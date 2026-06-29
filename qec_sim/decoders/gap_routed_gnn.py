"""Gap-routed GNN decoder (inference-only router).

Mirrors qec_sim.decoders.gap_routed_cnn.GapRoutedCNN3D but uses the GNN strong
stage (GapRoutedGNN) that consumes E_0 / E_1 byproducts natively as edge
features. Training is performed via the YAML pipeline:

    python main.py -c configs/gap_routed_gnn_d5.yaml -m train

Decoding flow (per shot):
    syndrome → MWPM augmented (logical=0/1) → w_0, w_1, E_0, E_1
            → gap = |w_1 - w_0|;  mwpm_pred = argmin(w_0, w_1)
            if gap ≥ τ: return mwpm_pred
            else:       return GNN(node_x, edge_feat_undir, scalars)
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


def _build_edge_lookup_aug(EC):
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


def _build_decoding_graph(EC, n_det: int):
    """Return (n_undir_e, undir_edges, undir_to_efidx, edge_index, undir_idx_for_dir).

    Boundary endpoint is mapped to a virtual node at index n_det.
    """
    EC_for_graph = EC.tocsc()
    n_edges_total = EC_for_graph.shape[1]
    VIRTUAL = n_det

    undir_edges = []
    undir_to_efidx = []
    for e_idx in range(n_edges_total):
        rows = EC_for_graph.getcol(e_idx).nonzero()[0]
        if rows.size == 1:
            a, b = int(rows[0]), VIRTUAL
        elif rows.size == 2:
            a, b = int(rows[0]), int(rows[1])
            if a > b:
                a, b = b, a
        else:
            continue
        undir_edges.append((a, b))
        undir_to_efidx.append(e_idx)

    n_undir = len(undir_edges)
    src_dir = np.zeros(2 * n_undir, dtype=np.int64)
    dst_dir = np.zeros(2 * n_undir, dtype=np.int64)
    undir_idx_for_dir = np.zeros(2 * n_undir, dtype=np.int64)
    for i, (a, b) in enumerate(undir_edges):
        src_dir[2 * i],     dst_dir[2 * i],     undir_idx_for_dir[2 * i]     = a, b, i
        src_dir[2 * i + 1], dst_dir[2 * i + 1], undir_idx_for_dir[2 * i + 1] = b, a, i

    edge_index = np.stack([src_dir, dst_dir], axis=0)
    return n_undir, undir_edges, np.array(undir_to_efidx), edge_index, undir_idx_for_dir


@register_decoder("gap_routed_gnn")
class GapRoutedGNNDecoder(BaseDecoder):
    """Inference-time router: MWPM(gap) → either MWPM or pre-trained GNN."""

    def __init__(
        self,
        error_model: stim.DetectorErrorModel,
        circuit: Optional[stim.Circuit] = None,
        tau: float = 2.0,
        device: str = "cuda",
        gnn_weights_path: Optional[str] = None,
        # model hyperparams must match the trained checkpoint
        d_hidden: int = 64,
        n_layers: int = 3,
        msg_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.2,
        pool: str = "mean",
        # CUDA Graphs (inference acceleration): capture forward once, replay
        # with new tensor data → ~10× lower kernel launch overhead for the
        # GNN (which has 30+ small kernels per forward).
        use_cuda_graph: bool = False,
        graph_batch_size: int = 512,
        **kwargs,
    ):
        super().__init__(circuit=circuit)
        if circuit is None:
            raise ValueError("GapRoutedGNNDecoder needs the stim circuit.")
        self.tau = float(tau)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_det = circuit.num_detectors
        # CUDA graphs only useful on CUDA, and only safe in eval mode.
        self._use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self._graph_batch_size = int(graph_batch_size)
        self._cuda_graph = None
        self._graph_static_in = None
        self._graph_static_out = None

        # ─── MWPM aug (mirror preprocessor) ───
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
        self._edge_lookup_aug = _build_edge_lookup_aug(EC_aug)

        # ─── decoding-graph topology ───
        (n_undir, undir_edges, undir_to_efidx,
         edge_index, undir_idx_for_dir) = _build_decoding_graph(EC, self.n_det)
        self.n_undir_e = n_undir
        self.n_node = self.n_det + 1
        VIRTUAL = self.n_det

        # static normalized edge weight (matches preprocessor)
        edge_w_undir = w_static[undir_to_efidx]
        w_mean = float(edge_w_undir.mean())
        w_std = float(edge_w_undir.std() + 1e-9)
        edge_w_norm = ((edge_w_undir - w_mean) / w_std).astype(np.float32)

        # (a_mapped, b_mapped) → undirected edge idx
        self._endpoint_to_undir_idx = {}
        for i, (a, b) in enumerate(undir_edges):
            self._endpoint_to_undir_idx[(a, b)] = i
            self._endpoint_to_undir_idx[(b, a)] = i

        # ─── build GNN model ───
        from qec_sim.models.gap_routed_gnn import GapRoutedGNN
        self.model = GapRoutedGNN(
            num_observables=1,
            n_node=self.n_node,
            n_undir_e=self.n_undir_e,
            d_edge=3,
            n_scalars=6,
            edge_index_np=edge_index,
            edge_w_norm_np=edge_w_norm,
            undir_idx_for_dir_np=undir_idx_for_dir,
            d_hidden=d_hidden,
            n_layers=n_layers,
            msg_hidden=msg_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
            pool=pool,
        ).to(self.device)
        self.model.eval()

        # buffers for inference packing
        self._edge_w_norm_torch = torch.from_numpy(edge_w_norm).to(self.device)

        self._loaded = False
        if gnn_weights_path is not None:
            self.load_gnn_weights(gnn_weights_path)

    # ----------------- weights I/O -----------------
    def load_gnn_weights(self, path: str | Path):
        state = torch.load(str(path), map_location=self.device)
        if isinstance(state, dict) and "core_model" in state:
            state = state["core_model"]
        elif isinstance(state, dict) and any(k.startswith("core_model.") for k in state):
            state = {k.replace("core_model.", ""): v
                     for k, v in state.items() if k.startswith("core_model.")}
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self._loaded = True

    # ----------------- per-shot feature extraction -----------------
    def _features_for_shot(self, syn_np: np.ndarray):
        n_det = self.n_det
        VIRTUAL = self.n_det
        n_undir = self.n_undir_e

        syn0 = np.concatenate([syn_np, [0]]).astype(np.uint8)
        syn1 = np.concatenate([syn_np, [1]]).astype(np.uint8)
        e0 = self._M_aug.decode_to_edges_array(syn0)
        e1 = self._M_aug.decode_to_edges_array(syn1)

        edge_in_E0 = np.zeros(n_undir, dtype=np.float32)
        edge_in_E1 = np.zeros(n_undir, dtype=np.float32)
        w0 = 0.0
        for a, b in e0:
            a, b = int(a), int(b)
            eid = self._edge_lookup_aug.get((a, b))
            if eid is not None:
                w0 += float(self._edge_weights[eid])
            a_map = a if 0 <= a < n_det else VIRTUAL
            b_map = b if 0 <= b < n_det else VIRTUAL
            u_idx = self._endpoint_to_undir_idx.get((a_map, b_map))
            if u_idx is not None:
                edge_in_E0[u_idx] = 1.0
        w1 = 0.0
        for a, b in e1:
            a, b = int(a), int(b)
            eid = self._edge_lookup_aug.get((a, b))
            if eid is not None:
                w1 += float(self._edge_weights[eid])
            a_map = a if 0 <= a < n_det else VIRTUAL
            b_map = b if 0 <= b < n_det else VIRTUAL
            u_idx = self._endpoint_to_undir_idx.get((a_map, b_map))
            if u_idx is not None:
                edge_in_E1[u_idx] = 1.0
        gap = abs(w0 - w1)
        if w0 <= w1:
            mwpm_pred, weight_sum, n_matched = 0, w0, len(e0)
        else:
            mwpm_pred, weight_sum, n_matched = 1, w1, len(e1)
        return mwpm_pred, gap, w0, w1, weight_sum, n_matched, edge_in_E0, edge_in_E1

    def _pack_features_batch(self, dets: np.ndarray):
        """For (N, n_det) syndromes → (mwpm_pred, gap, packed_tensor)."""
        N, n_det = dets.shape[0], self.n_det
        n_undir = self.n_undir_e
        d_e = 3

        mwpm_pred = np.zeros(N, dtype=np.uint8)
        gap = np.zeros(N, dtype=np.float64)
        scalars = np.zeros((N, 6), dtype=np.float32)
        edge_E0 = np.zeros((N, n_undir), dtype=np.float32)
        edge_E1 = np.zeros((N, n_undir), dtype=np.float32)

        for i in range(N):
            (p, g, w0, w1, ws, nm, e0_v, e1_v) = self._features_for_shot(dets[i])
            mwpm_pred[i] = p
            gap[i] = g
            scalars[i] = [g, w0, w1, ws, nm / max(1, n_det), float(p)]
            edge_E0[i] = e0_v
            edge_E1[i] = e1_v

        # Build flat tensor: [node_x (N, n_node)] + [edge_feat (N, n_undir, 3) flat] + [scalars (N, 6)]
        node_x = np.zeros((N, self.n_node), dtype=np.float32)
        node_x[:, :n_det] = dets.astype(np.float32)

        edge_w_norm = self._edge_w_norm_torch.cpu().numpy()      # (n_undir,)
        edge_feat = np.stack([
            np.broadcast_to(edge_w_norm[None, :], (N, n_undir)).copy(),
            edge_E0,
            edge_E1,
        ], axis=-1).astype(np.float32)                            # (N, n_undir, 3)

        flat = np.concatenate([
            node_x,
            edge_feat.reshape(N, -1),
            scalars,
        ], axis=-1).astype(np.float32)
        return mwpm_pred, gap, flat

    def _build_cuda_graph(self, feat_dim: int):
        """Capture the model forward into a CUDA graph once. Subsequent
        replays skip ~30 kernel launch overheads per shot.

        Requirements (PyTorch CUDA Graphs):
          - model.eval() (no dropout randomness)
          - static input/output buffers
          - 3+ warmup iterations on a side stream before capture (initializes
            the caching allocator + cuBLAS workspace)
        """
        B = self._graph_batch_size
        # static buffers
        self._graph_static_in = torch.zeros(B, feat_dim, device=self.device)
        # warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model(self._graph_static_in)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_static_out = self.model(self._graph_static_in)

    @torch.no_grad()
    def _gnn_predict(self, feat: np.ndarray, batch: int = 512):
        N = feat.shape[0]
        if N == 0:
            return np.empty(0, dtype=np.uint8)

        if self._use_cuda_graph:
            B = self._graph_batch_size
            if self._cuda_graph is None:
                self._build_cuda_graph(feat_dim=feat.shape[1])
            preds = np.empty(N, dtype=np.uint8)
            feat_t = torch.from_numpy(feat).to(self.device, non_blocking=True)
            for i in range(0, N, B):
                end = min(i + B, N)
                n_real = end - i
                # copy into static buffer; pad the rest with zeros (results discarded)
                self._graph_static_in[:n_real].copy_(feat_t[i:end])
                if n_real < B:
                    self._graph_static_in[n_real:].zero_()
                self._cuda_graph.replay()
                logits = self._graph_static_out[:n_real].cpu().numpy().flatten()
                preds[i:end] = (logits > 0).astype(np.uint8)
            return preds

        # eager fallback
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
                gnn_pred = self._gnn_predict(feat[routed])
                out[routed] = gnn_pred
        return out[:, None].astype(bool)

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        preds = self.decode_batch(syndrome[np.newaxis])
        return {
            "logical_error": preds[0].tolist(),
            "corrected_fault_ids": [],
        }
