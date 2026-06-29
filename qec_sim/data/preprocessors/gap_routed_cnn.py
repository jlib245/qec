# qec_sim/data/preprocessors/gap_routed_cnn.py
import torch
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor


# ─────────────────────────────────────────────────────────────────────────
# GapRoutedCNNPreprocessor
# ─────────────────────────────────────────────────────────────────────────
# Per-sample MWPM (standard + 2 augmented) → 4-channel grid + 6 scalars.
# Designed for the GapRoutedCNN model (qec_sim.models.gap_routed_cnn).
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("gap_routed_cnn")
class GapRoutedCNNPreprocessor(BasePreprocessor):
    """Compute MWPM features per syndrome and pack into a flat tensor.

    Grid channels (per-detector → placed on (T, H, W) grid):
      0  syndrome (detection events)
      1  edge_diff_mask           — endpoints of (E_0 △ E_1)
      2  w0_mask                  — per-detector E_0 edge weight sum
      3  w1_mask                  — per-detector E_1 edge weight sum

    Scalar features:
      gap, w_0, w_1, weight_sum, n_matched_norm, mwpm_pred
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        import numpy as np
        import pymatching
        from beliefmatching import detector_error_model_to_check_matrices
        from scipy import sparse

        if circuit is None:
            raise ValueError("GapRoutedCNNPreprocessor requires the stim circuit.")
        self._required_keys = ["syndromes"]
        self.num_detectors = circuit.num_detectors

        # MWPM (augmented only) — built from the DEM of this circuit.
        # Standard MWPM call is omitted: argmin(w_0, w_1) reproduces mwpm_pred
        # and min(w_0, w_1) reproduces its weight sum bit-identically.
        # For multi-p training the SimulatorPool samples mixed noise but we use
        # a single MWPM (built from this circuit's DEM, typically the median p).
        dem = circuit.detector_error_model(decompose_errors=True)
        mtx = detector_error_model_to_check_matrices(dem)
        priors = np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors)
        w_static = -np.log(np.clip(priors, 1e-14, 1 - 1e-14)).astype(np.float64)
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
        self._edge_lookup_aug = self._build_edge_lookup(EC_aug)

        # detector grid coordinates
        raw = circuit.get_detector_coordinates()
        coords = np.array([raw[i] for i in range(self.num_detectors)])
        xs = sorted(set(coords[:, 0]));  x2i = {v: i for i, v in enumerate(xs)}
        ys = sorted(set(coords[:, 1]));  y2i = {v: i for i, v in enumerate(ys)}
        ts = sorted(set(coords[:, 2]));  t2i = {v: i for i, v in enumerate(ts)}
        det_idx = np.array([
            [t2i[coords[i, 2]], y2i[coords[i, 1]], x2i[coords[i, 0]]]
            for i in range(self.num_detectors)
        ], dtype=np.int64)
        self.T, self.H, self.W = len(ts), len(ys), len(xs)
        self.n_grid_channels = 4
        self.n_scalars = 6
        self.grid_dim = self.n_grid_channels * self.T * self.H * self.W

        # flat index per detector for scatter
        flat_idx = (det_idx[:, 0] * self.H * self.W
                    + det_idx[:, 1] * self.W
                    + det_idx[:, 2])
        self.register_buffer(
            "_flat_idx", torch.from_numpy(flat_idx).long(), persistent=False
        )

    # ----- helpers -----
    @staticmethod
    def _build_edge_lookup(EC):
        from scipy import sparse
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

    # ----- interface -----
    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": self.n_grid_channels,
                "n_scalars": self.n_scalars}

    def cpu_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Batched MWPM feature extraction.

        batch["syndromes"]: tensor or array (B, n_det). Per-shot 2 augmented
        MWPM runs (syndrome appended with 0/1); returns 4 per-detector channel
        batches + 6 scalars. mwpm_pred / weight_sum derived from argmin(w_0,w_1).
        """
        import numpy as np
        syn = batch["syndromes"]
        if isinstance(syn, torch.Tensor):
            syn_np = syn.cpu().numpy()
        else:
            syn_np = np.asarray(syn)
        syn_np = syn_np.astype(np.uint8)
        if syn_np.ndim == 1:
            syn_np = syn_np[None, :]
        B, n_det = syn_np.shape

        ch_syn  = syn_np.astype(np.float32)
        ch_diff = np.zeros((B, n_det), dtype=np.float32)
        ch_w0   = np.zeros((B, n_det), dtype=np.float32)
        ch_w1   = np.zeros((B, n_det), dtype=np.float32)
        scalars = np.zeros((B, 6), dtype=np.float32)

        for i in range(B):
            s = syn_np[i]
            syn0 = np.concatenate([s, [0]]).astype(np.uint8)
            syn1 = np.concatenate([s, [1]]).astype(np.uint8)
            e0 = self._M_aug.decode_to_edges_array(syn0)
            e1 = self._M_aug.decode_to_edges_array(syn1)

            set0, set1 = set(), set()
            w0 = 0.0
            for a, b in e0:
                a, b = int(a), int(b)
                key = (a, b) if a <= b else (b, a)
                set0.add(key)
                eid = self._edge_lookup_aug.get((a, b))
                w = float(self._edge_weights[eid]) if eid is not None else 0.0
                w0 += w
                if 0 <= a < n_det: ch_w0[i, a] += w
                if 0 <= b < n_det: ch_w0[i, b] += w
            w1 = 0.0
            for a, b in e1:
                a, b = int(a), int(b)
                key = (a, b) if a <= b else (b, a)
                set1.add(key)
                eid = self._edge_lookup_aug.get((a, b))
                w = float(self._edge_weights[eid]) if eid is not None else 0.0
                w1 += w
                if 0 <= a < n_det: ch_w1[i, a] += w
                if 0 <= b < n_det: ch_w1[i, b] += w
            for (a, b) in (set0 ^ set1):
                if 0 <= a < n_det: ch_diff[i, a] = 1.0
                if 0 <= b < n_det: ch_diff[i, b] = 1.0

            gap = abs(w0 - w1)
            if w0 <= w1:
                mwpm_pred, weight_sum, n_matched = 0, w0, len(e0)
            else:
                mwpm_pred, weight_sum, n_matched = 1, w1, len(e1)
            scalars[i] = [gap, w0, w1, weight_sum,
                          n_matched / max(1, n_det),
                          float(mwpm_pred)]

        return {
            "ch_syn":  torch.from_numpy(ch_syn),
            "ch_diff": torch.from_numpy(ch_diff),
            "ch_w0":   torch.from_numpy(ch_w0),
            "ch_w1":   torch.from_numpy(ch_w1),
            "scalars": torch.from_numpy(scalars),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack per-detector channels onto (B, 4, T, H, W) grid, concat with scalars."""
        ch_syn  = batch_data["ch_syn"]
        ch_diff = batch_data["ch_diff"]
        ch_w0   = batch_data["ch_w0"]
        ch_w1   = batch_data["ch_w1"]
        scalars = batch_data["scalars"]
        B = ch_syn.size(0)
        device = ch_syn.device

        # scatter channels onto flat (T*H*W) grid
        grid = torch.zeros(B, self.n_grid_channels, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        grid[:, 1, flat_idx] = ch_diff.float()
        grid[:, 2, flat_idx] = ch_w0.float()
        grid[:, 3, flat_idx] = ch_w1.float()

        return torch.cat([grid.reshape(B, -1), scalars.float()], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# GapRoutedCNNFilteredPreprocessor — gap-filtered training (fine-tune on
# routed sub-distribution, gap < threshold)
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("gap_routed_cnn_filtered")
class GapRoutedCNNFilteredPreprocessor(GapRoutedCNNPreprocessor):
    """Same as GapRoutedCNNPreprocessor but drops samples with gap >= threshold.

    Used for fine-tuning the CNN on the routed sub-distribution only
    (matches deployment-time CNN input). Inserts `_keep_mask` into the
    cpu_transform output; the dataset applies the mask to features + labels.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, gap_threshold: float = 2.0, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.gap_threshold = float(gap_threshold)

    def cpu_transform(self, batch):
        out = super().cpu_transform(batch)
        gaps = out["scalars"][:, 0]
        out["_keep_mask"] = (gaps < self.gap_threshold)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Syndrome3DPreprocessor — generic 3D syndrome grid (1 channel, no scalars)
# ─────────────────────────────────────────────────────────────────────────
# Detector coordinates from the circuit are arranged into a (T, H, W) grid.
# Only the syndrome (detection events) is placed; no DEM-derived per-shot
# features. Useful as a baseline for 3D CNN syndrome decoders.
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("syndrome_3d")
class Syndrome3DPreprocessor(BasePreprocessor):
    """Generic 3D syndrome grid preprocessor (1 grid channel, 0 scalars).

    Same (T, H, W) detector layout as GapRoutedCNNPreprocessor — usable as a
    baseline for any 3D CNN model that consumes the syndrome volume directly.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        import numpy as np
        if circuit is None:
            raise ValueError("Syndrome3DPreprocessor requires the stim circuit.")
        self._required_keys = ["syndromes"]
        self.num_detectors = circuit.num_detectors

        raw = circuit.get_detector_coordinates()
        coords = np.array([raw[i] for i in range(self.num_detectors)])
        xs = sorted(set(coords[:, 0]));  x2i = {v: i for i, v in enumerate(xs)}
        ys = sorted(set(coords[:, 1]));  y2i = {v: i for i, v in enumerate(ys)}
        ts = sorted(set(coords[:, 2]));  t2i = {v: i for i, v in enumerate(ts)}
        det_idx = np.array([
            [t2i[coords[i, 2]], y2i[coords[i, 1]], x2i[coords[i, 0]]]
            for i in range(self.num_detectors)
        ], dtype=np.int64)
        self.T, self.H, self.W = len(ts), len(ys), len(xs)
        self.n_grid_channels = 1
        self.n_scalars = 0
        self.grid_dim = self.n_grid_channels * self.T * self.H * self.W

        flat_idx = (det_idx[:, 0] * self.H * self.W
                    + det_idx[:, 1] * self.W
                    + det_idx[:, 2])
        self.register_buffer(
            "_flat_idx", torch.from_numpy(flat_idx).long(), persistent=False
        )

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": self.n_grid_channels,
                "n_scalars": self.n_scalars}

    def cpu_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        syn = batch["syndromes"]
        if not isinstance(syn, torch.Tensor):
            syn = torch.as_tensor(syn)
        return {"ch_syn": syn.float()}

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ch_syn = batch_data["ch_syn"]
        B = ch_syn.size(0)
        device = ch_syn.device

        grid = torch.zeros(B, self.n_grid_channels, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        return grid.reshape(B, -1)


# ─────────────────────────────────────────────────────────────────────────
# Syndrome3DFilteredPreprocessor — syndrome-only input + gap-based filtering
# ─────────────────────────────────────────────────────────────────────────
# Computes MWPM/gap internally (for filtering) but only outputs the syndrome
# volume. Used for fine-tuning the syndrome-only baseline on the routed
# sub-distribution (gap < threshold).
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("syndrome_3d_filtered")
class Syndrome3DFilteredPreprocessor(GapRoutedCNNPreprocessor):
    """Syndrome-only preprocessor with gap-based filtering.

    Inherits MWPM setup from GapRoutedCNNPreprocessor (used only for gap
    computation). Output is 1 channel (syndrome) + 0 scalars + `_keep_mask`.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, gap_threshold: float = 2.0, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.gap_threshold = float(gap_threshold)
        # Override channel counts — model sees only the syndrome.
        self.n_grid_channels = 1
        self.n_scalars = 0
        self.grid_dim = 1 * self.T * self.H * self.W

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": 1, "n_scalars": 0}

    def cpu_transform(self, batch):
        full = super().cpu_transform(batch)  # has MWPM byproducts + gap
        gaps = full["scalars"][:, 0]
        return {
            "ch_syn":     full["ch_syn"],
            "_keep_mask": (gaps < self.gap_threshold),
        }

    def gpu_transform(self, batch_data):
        ch_syn = batch_data["ch_syn"]
        B = ch_syn.size(0)
        device = ch_syn.device
        grid = torch.zeros(B, 1, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        return grid.reshape(B, -1)


# ─────────────────────────────────────────────────────────────────────────
# Cached variants — share the .npz produced by scripts/build_routed_cache.py
# with the GNN family. CNN's per-detector ch_diff/ch_w0/ch_w1 channels are
# derived from edge_in_E0/E1 at load time (vectorized scatter onto detector
# array; ~µs per sample, dominated by I/O not compute).
# ─────────────────────────────────────────────────────────────────────────


@register_preprocessor("cached_gap_routed_cnn")
class CachedGapRoutedCNNPreprocessor(GapRoutedCNNPreprocessor):
    """Offline-cached GapRoutedCNN preprocessor.

    Reads `syndromes, edge_in_E0, edge_in_E1, scalars` from shared cache .npz
    and derives the per-detector ch_diff/ch_w0/ch_w1 channels from edges via
    static edge endpoints + edge weights. gpu_transform inherits the parent's
    grid-scatter logic (channels packed onto (T, H, W) volume).
    """

    def __init__(self, circuit, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        # Build undirected-edge topology mirroring GapRoutedGNNPreprocessor so
        # cache .npz `edge_in_E0/E1` indices align bit-for-bit.
        import numpy as np
        from scipy import sparse
        from beliefmatching import detector_error_model_to_check_matrices

        dem = circuit.detector_error_model(decompose_errors=True)
        mtx = detector_error_model_to_check_matrices(dem)
        priors = np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors)
        w_static = -np.log(np.clip(priors, 1e-14, 1 - 1e-14)).astype(np.float32)
        EC = sparse.csc_matrix(mtx.edge_check_matrix).tocsc()
        n_edges_total = EC.shape[1]
        VIRTUAL = self.num_detectors  # boundary endpoint marker

        endpoints_a, endpoints_b, edge_w_list = [], [], []
        for e_idx in range(n_edges_total):
            rows = EC.getcol(e_idx).nonzero()[0]
            if rows.size == 1:
                a, b = int(rows[0]), VIRTUAL
            elif rows.size == 2:
                a, b = int(rows[0]), int(rows[1])
                if a > b:
                    a, b = b, a
            else:
                continue
            endpoints_a.append(a)
            endpoints_b.append(b)
            edge_w_list.append(w_static[e_idx])

        self.n_undir_e = len(endpoints_a)
        self.register_buffer(
            "_edge_endpoint_a",
            torch.tensor(endpoints_a, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_edge_endpoint_b",
            torch.tensor(endpoints_b, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_edge_w_raw",
            torch.tensor(edge_w_list, dtype=torch.float32),
            persistent=False,
        )

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes", "edge_in_E0", "edge_in_E1", "scalars"]

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Pass-through only. The per-detector channel derivation needs the
        # static edge-endpoint buffers, which the wrapper moves to GPU when
        # `.to(device)` is called — so the scatter MUST run in gpu_transform
        # (DataLoader workers cannot touch CUDA tensors).
        return {
            "ch_syn":     sample["syndromes"].float(),
            "edge_in_E0": sample["edge_in_E0"].float(),
            "edge_in_E1": sample["edge_in_E1"].float(),
            "scalars":    sample["scalars"].float(),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Derive per-detector ch_diff/ch_w0/ch_w1 from edges (batched, on GPU).
        ch_syn = batch_data["ch_syn"]
        e0 = batch_data["edge_in_E0"]
        e1 = batch_data["edge_in_E1"]
        scalars = batch_data["scalars"]
        B = ch_syn.size(0)
        device = ch_syn.device
        n_det = self.num_detectors

        a = self._edge_endpoint_a.to(device)        # (n_undir_e,)
        b = self._edge_endpoint_b.to(device)
        w = self._edge_w_raw.to(device)             # (n_undir_e,)

        ch_w0 = torch.zeros(B, n_det, device=device, dtype=torch.float32)
        ch_w1 = torch.zeros(B, n_det, device=device, dtype=torch.float32)
        ch_diff = torch.zeros(B, n_det, device=device, dtype=torch.float32)

        # Mask edges incident to a real detector (vs virtual boundary = n_det)
        valid_a = a < n_det
        valid_b = b < n_det
        a_valid = a[valid_a]
        b_valid = b[valid_b]
        a_idx = a_valid.view(1, -1).expand(B, -1)   # (B, n_valid_a)
        b_idx = b_valid.view(1, -1).expand(B, -1)

        # Edge contributions weighted by w_e for E_0 / E_1
        contrib_e0 = e0 * w.view(1, -1)             # (B, n_undir_e)
        contrib_e1 = e1 * w.view(1, -1)
        ch_w0.scatter_add_(1, a_idx, contrib_e0[:, valid_a])
        ch_w0.scatter_add_(1, b_idx, contrib_e0[:, valid_b])
        ch_w1.scatter_add_(1, a_idx, contrib_e1[:, valid_a])
        ch_w1.scatter_add_(1, b_idx, contrib_e1[:, valid_b])

        # ch_diff: detector incident to any edge in E_0 △ E_1
        diff = (e0.bool() ^ e1.bool()).float()
        ch_diff.scatter_reduce_(1, a_idx, diff[:, valid_a], reduce="amax",
                                 include_self=True)
        ch_diff.scatter_reduce_(1, b_idx, diff[:, valid_b], reduce="amax",
                                 include_self=True)

        # Hand off to parent's grid scatter (4 channels onto T×H×W)
        return super().gpu_transform({
            "ch_syn":  ch_syn,
            "ch_diff": ch_diff,
            "ch_w0":   ch_w0,
            "ch_w1":   ch_w1,
            "scalars": scalars,
        })


@register_preprocessor("cached_syndrome_3d")
class CachedSyndrome3DPreprocessor(Syndrome3DPreprocessor):
    """Cached syndrome-only 3D CNN ablation — reads only `syndromes` from the
    shared cache .npz."""

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes"]

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"ch_syn": sample["syndromes"].float()}
