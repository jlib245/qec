"""Gap-routed CNN3D decoder (inference-only router).

This decoder wraps a pre-trained CNN (qec_sim.models.gap_routed_cnn.GapRoutedCNN)
with MWPM-based routing. Training is performed via the YAML training pipeline:

    python main.py -c configs/gap_routed_cnn_d5.yaml -m train

The trained weights (best_model.pth) are then passed to this decoder for
deployment-style evaluation.

Decoding flow:
    syndrome → MWPM augmented (logical=0/1) → w_0, w_1, edge sets E_0/E_1
            → gap = |w_1 - w_0|;  mwpm_pred = argmin(w_0, w_1)
            if gap ≥ τ: return mwpm_pred
            else:       return CNN(grid, scalars)

Routing happens at inference only; the CNN itself is trained on the full data
distribution and learns to imitate MWPM when gap is large.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pymatching
import stim
import torch
from beliefmatching import detector_error_model_to_check_matrices
from scipy import sparse

from .base import BaseDecoder
from .registry import register_decoder


def _build_edge_lookup(EC):
    if not sparse.issparse(EC):
        EC = sparse.csc_matrix(EC)
    EC = EC.tocsc()
    n_edge = EC.shape[1]
    lookup = {}
    for e in range(n_edge):
        rows = EC.getcol(e).nonzero()[0]
        if rows.size == 1:
            lookup[(int(rows[0]), -1)] = e
            lookup[(-1, int(rows[0]))] = e
        elif rows.size == 2:
            a, b = int(rows[0]), int(rows[1])
            lookup[(a, b)] = e
            lookup[(b, a)] = e
    return lookup


def _build_detector_map(circ: stim.Circuit):
    n = circ.num_detectors
    raw = circ.get_detector_coordinates()
    coords = np.array([raw[i] for i in range(n)])
    xs = sorted(set(coords[:, 0])); x2i = {v: i for i, v in enumerate(xs)}
    ys = sorted(set(coords[:, 1])); y2i = {v: i for i, v in enumerate(ys)}
    ts = sorted(set(coords[:, 2])); t2i = {v: i for i, v in enumerate(ts)}
    idx = np.array([
        [t2i[coords[i, 2]], y2i[coords[i, 1]], x2i[coords[i, 0]]]
        for i in range(n)
    ], dtype=np.int64)
    return idx, len(ts), len(ys), len(xs)


@register_decoder("gap_routed_cnn3d")
class GapRoutedCNN3D(BaseDecoder):
    """Inference-time router: MWPM(gap) → either MWPM or pre-trained CNN3D."""

    def __init__(
        self,
        error_model: stim.DetectorErrorModel,
        circuit: Optional[stim.Circuit] = None,
        tau: float = 2.0,
        device: str = "cuda",
        cnn_weights_path: Optional[str] = None,
        # CNN hyperparams must match the trained model
        conv_channels=(32, 64, 128),
        hidden: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(circuit=circuit)
        if circuit is None:
            raise ValueError("GapRoutedCNN3D needs the stim circuit "
                             "(detector coordinates).")
        self.tau = float(tau)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ─── MWPM (augmented only) ───
        # Standard MWPM call is omitted: argmin(w_0, w_1) reproduces
        # mwpm_pred and min(w_0, w_1) reproduces its weight sum bit-identically
        # (Phase 0 verification, 25k shots across d=5/7 × p=0.005/0.008).
        mtx = detector_error_model_to_check_matrices(error_model)
        w_static = -np.log(np.clip(
            np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors),
            1e-14, 1 - 1e-14,
        )).astype(np.float64)
        self._edge_weights = w_static
        EC = sparse.csc_matrix(mtx.edge_check_matrix)
        EO = sparse.csr_matrix(mtx.edge_observables_matrix)
        obs_row = sparse.csr_matrix(EO.toarray()[0:1])
        EC_aug = sparse.vstack([EC, obs_row]).tocsc()
        self._M_aug = pymatching.Matching.from_check_matrix(
            EC_aug, weights=w_static,
            faults_matrix=EO,
            use_virtual_boundary_node=True,
        )
        self._edge_lookup_aug = _build_edge_lookup(EC_aug)

        # ─── detector grid ───
        self._det_idx, self.T, self.H, self.W = _build_detector_map(circuit)
        self.n_det = circuit.num_detectors
        self._flat_idx = (self._det_idx[:, 0] * self.H * self.W
                          + self._det_idx[:, 1] * self.W
                          + self._det_idx[:, 2])

        # ─── CNN model (matching trained architecture) ───
        from qec_sim.models.gap_routed_cnn import GapRoutedCNN
        self.model = GapRoutedCNN(
            num_observables=1,
            T=self.T, H=self.H, W=self.W,
            n_grid_channels=4, n_scalars=6,
            conv_channels=conv_channels,
            hidden=hidden, dropout=dropout,
        ).to(self.device)
        self.model.eval()
        self._loaded = False
        if cnn_weights_path is not None:
            self.load_cnn_weights(cnn_weights_path)

    # ----------------- weights I/O -----------------
    def load_cnn_weights(self, path: str | Path):
        state = torch.load(str(path), map_location=self.device)
        # support both bare state_dict and a wrapper dict
        if isinstance(state, dict) and "core_model" in state:
            state = state["core_model"]
        elif isinstance(state, dict) and any(k.startswith("core_model.") for k in state):
            state = {k.replace("core_model.", ""): v
                     for k, v in state.items() if k.startswith("core_model.")}
        # tolerate "model_state" keyed save (from older scripts)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self._loaded = True

    # ----------------- per-shot feature extraction -----------------
    def _features_for_shot(self, syn_np: np.ndarray):
        n_det = self.n_det

        syn0 = np.concatenate([syn_np, [0]]).astype(np.uint8)
        syn1 = np.concatenate([syn_np, [1]]).astype(np.uint8)
        e0 = self._M_aug.decode_to_edges_array(syn0)
        e1 = self._M_aug.decode_to_edges_array(syn1)

        ch_w0 = np.zeros(n_det, dtype=np.float32)
        ch_w1 = np.zeros(n_det, dtype=np.float32)
        set0, set1 = set(), set()
        w0 = 0.0
        for a, b in e0:
            a, b = int(a), int(b)
            key = (a, b) if a <= b else (b, a)
            set0.add(key)
            eid = self._edge_lookup_aug.get((a, b))
            w = float(self._edge_weights[eid]) if eid is not None else 0.0
            w0 += w
            if 0 <= a < n_det: ch_w0[a] += w
            if 0 <= b < n_det: ch_w0[b] += w
        w1 = 0.0
        for a, b in e1:
            a, b = int(a), int(b)
            key = (a, b) if a <= b else (b, a)
            set1.add(key)
            eid = self._edge_lookup_aug.get((a, b))
            w = float(self._edge_weights[eid]) if eid is not None else 0.0
            w1 += w
            if 0 <= a < n_det: ch_w1[a] += w
            if 0 <= b < n_det: ch_w1[b] += w
        ch_diff = np.zeros(n_det, dtype=np.float32)
        for (a, b) in (set0 ^ set1):
            if 0 <= a < n_det: ch_diff[a] = 1.0
            if 0 <= b < n_det: ch_diff[b] = 1.0
        gap = abs(w0 - w1)

        if w0 <= w1:
            mwpm_pred, weight_sum, n_matched = 0, w0, len(e0)
        else:
            mwpm_pred, weight_sum, n_matched = 1, w1, len(e1)
        return mwpm_pred, gap, w0, w1, weight_sum, n_matched, ch_diff, ch_w0, ch_w1

    def _pack_features_batch(self, dets: np.ndarray):
        """For (N, n_det) syndromes → (mwpm_pred, gap, packed_tensor)."""
        N, n_det = dets.shape[0], self.n_det
        mwpm_pred = np.zeros(N, dtype=np.uint8)
        gap = np.zeros(N, dtype=np.float64)
        scalars = np.zeros((N, 6), dtype=np.float32)
        ch_syn  = dets.astype(np.float32)
        ch_diff = np.zeros((N, n_det), dtype=np.float32)
        ch_w0   = np.zeros((N, n_det), dtype=np.float32)
        ch_w1   = np.zeros((N, n_det), dtype=np.float32)

        for i in range(N):
            (p, g, w0, w1, ws, nm, d, m0, m1) = self._features_for_shot(dets[i])
            mwpm_pred[i] = p
            gap[i] = g
            scalars[i] = [g, w0, w1, ws, nm / max(1, n_det), float(p)]
            ch_diff[i] = d
            ch_w0[i]   = m0
            ch_w1[i]   = m1

        # build grid (N, 4, T*H*W) then flatten + concat scalars
        grid = np.zeros((N, 4, self.T * self.H * self.W), dtype=np.float32)
        grid[:, 0, self._flat_idx] = ch_syn
        grid[:, 1, self._flat_idx] = ch_diff
        grid[:, 2, self._flat_idx] = ch_w0
        grid[:, 3, self._flat_idx] = ch_w1
        feat = np.concatenate([grid.reshape(N, -1), scalars], axis=1)
        return mwpm_pred, gap, feat

    @torch.no_grad()
    def _cnn_predict(self, feat: np.ndarray, batch: int = 512):
        x = torch.from_numpy(feat).to(self.device)
        outs = []
        for i in range(0, x.shape[0], batch):
            j = slice(i, min(i + batch, x.shape[0]))
            outs.append(self.model(x[j]))
        logits = torch.cat(outs).cpu().numpy().flatten()
        return (logits > 0).astype(np.uint8)

    # ----------------- BaseDecoder API -----------------
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        mwpm_pred, gap, feat = self._pack_features_batch(syndromes)
        out = mwpm_pred.copy()
        if self._loaded:
            routed = gap < self.tau
            if routed.any():
                cnn_pred = self._cnn_predict(feat[routed])
                out[routed] = cnn_pred
        return out[:, None].astype(bool)

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        preds = self.decode_batch(syndrome[np.newaxis])
        return {
            "logical_error": preds[0].tolist(),
            "corrected_fault_ids": [],
        }
