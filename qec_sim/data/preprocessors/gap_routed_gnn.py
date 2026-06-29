# qec_sim/data/preprocessors/gap_routed_gnn.py
import torch
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor



# ─────────────────────────────────────────────────────────────────────────
# GapRoutedGNNPreprocessor — graph-structured input for the GNN strong stage
# ─────────────────────────────────────────────────────────────────────────
# Per-shot MWPM (2 augmented) → per-EDGE binary indicators 1[e∈E_0], 1[e∈E_1]
# (no per-detector aggregation; CNN's grid-projection loss is avoided).
#
# Static graph topology:
#   n_node       = num_detectors + 1     (virtual boundary node at index n_det)
#   n_undir_e    = # undirected decoding-graph edges (from EC)
#   edge_index   shape (2, 2*n_undir_e)  bidirected
#   edge_w_norm  shape (n_undir_e,)      standardized -log p_e
#
# Per-shot edge features (3-dim): [w_e_norm, 1[e∈E_0], 1[e∈E_1]].
# Output of gpu_transform is a flat tensor:
#   [node_x (B, n_node)] + [edge_feat (B, n_undir_e * 3)] + [scalars (B, 6)]
# The GapRoutedGNN model unpacks this layout.
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("gap_routed_gnn")
class GapRoutedGNNPreprocessor(BasePreprocessor):
    """Per-shot MWPM aug → graph-structured features for GapRoutedGNN.

    Outputs per-edge binary indicators (1[e∈E_0], 1[e∈E_1]) and a static
    normalized edge weight w_e_norm. No per-detector aggregation.
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
            raise ValueError("GapRoutedGNNPreprocessor requires the stim circuit.")
        self._required_keys = ["syndromes"]
        self.num_detectors = circuit.num_detectors
        n_det = self.num_detectors

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
        # lookup uses EC_aug indices (matches pymatching.decode_to_edges output)
        self._edge_lookup_aug = self._build_edge_lookup_aug(EC_aug)

        # Build undirected edge list from the original EC (detector rows only).
        # EC columns ↔ EC_aug columns ↔ fault mechanisms (same indexing).
        EC_for_graph = EC.tocsc()
        n_edges_total = EC_for_graph.shape[1]
        VIRTUAL = n_det

        undir_edges = []        # canonical (a, b) with a <= b; boundary endpoint = VIRTUAL
        undir_to_efidx = []     # undirected idx → EC column (for w_static lookup)
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

        edge_w_undir = w_static[np.array(undir_to_efidx)]
        w_mean = float(edge_w_undir.mean())
        w_std = float(edge_w_undir.std() + 1e-9)
        edge_w_norm = ((edge_w_undir - w_mean) / w_std).astype(np.float32)

        # (a_mapped, b_mapped) → undirected edge idx (for per-shot E_0/E_1 marking)
        self._endpoint_to_undir_idx = {}
        for i, (a, b) in enumerate(undir_edges):
            self._endpoint_to_undir_idx[(a, b)] = i
            self._endpoint_to_undir_idx[(b, a)] = i

        self.n_det = n_det
        self.n_node = n_det + 1
        self.n_undir_e = n_undir
        self.d_edge = 3
        self.n_scalars = 6

        # numpy arrays — passed via get_model_kwargs to the model
        self._edge_index_np = np.stack([src_dir, dst_dir], axis=0)
        self._edge_w_norm_np = edge_w_norm
        self._undir_idx_for_dir_np = undir_idx_for_dir

        self.register_buffer(
            "_edge_w_norm", torch.from_numpy(edge_w_norm), persistent=False
        )

    @staticmethod
    def _build_edge_lookup_aug(EC):
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

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {
            "n_node": self.n_node,
            "n_undir_e": self.n_undir_e,
            "d_edge": self.d_edge,
            "n_scalars": self.n_scalars,
            "edge_index_np": self._edge_index_np,
            "edge_w_norm_np": self._edge_w_norm_np,
            "undir_idx_for_dir_np": self._undir_idx_for_dir_np,
        }

    def cpu_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
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
        n_undir = self.n_undir_e
        VIRTUAL = self.n_det

        ch_syn = syn_np.astype(np.float32)
        edge_in_E0 = np.zeros((B, n_undir), dtype=np.float32)
        edge_in_E1 = np.zeros((B, n_undir), dtype=np.float32)
        scalars = np.zeros((B, 6), dtype=np.float32)

        for i in range(B):
            s = syn_np[i]
            syn0 = np.concatenate([s, [0]]).astype(np.uint8)
            syn1 = np.concatenate([s, [1]]).astype(np.uint8)
            e0 = self._M_aug.decode_to_edges_array(syn0)
            e1 = self._M_aug.decode_to_edges_array(syn1)

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
                    edge_in_E0[i, u_idx] = 1.0

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
                    edge_in_E1[i, u_idx] = 1.0

            gap = abs(w0 - w1)
            if w0 <= w1:
                mwpm_pred, weight_sum, n_matched = 0, w0, len(e0)
            else:
                mwpm_pred, weight_sum, n_matched = 1, w1, len(e1)
            scalars[i] = [gap, w0, w1, weight_sum,
                          n_matched / max(1, n_det),
                          float(mwpm_pred)]

        return {
            "ch_syn":     torch.from_numpy(ch_syn),
            "edge_in_E0": torch.from_numpy(edge_in_E0),
            "edge_in_E1": torch.from_numpy(edge_in_E1),
            "scalars":    torch.from_numpy(scalars),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ch_syn = batch_data["ch_syn"]
        edge_in_E0 = batch_data["edge_in_E0"]
        edge_in_E1 = batch_data["edge_in_E1"]
        scalars = batch_data["scalars"]
        B = ch_syn.size(0)
        device = ch_syn.device

        # node features: detector syndrome + virtual node (always 0)
        node_x = torch.zeros(B, self.n_node, device=device, dtype=torch.float32)
        node_x[:, : self.n_det] = ch_syn.float()

        edge_w_norm = self._edge_w_norm.to(device)              # (n_undir,)
        edge_feat = torch.stack([
            edge_w_norm.unsqueeze(0).expand(B, -1),
            edge_in_E0.float(),
            edge_in_E1.float(),
        ], dim=-1)                                              # (B, n_undir, 3)

        flat = torch.cat([
            node_x,
            edge_feat.reshape(B, -1),
            scalars.float(),
        ], dim=-1)
        return flat


@register_preprocessor("gap_routed_gnn_filtered")
class GapRoutedGNNFilteredPreprocessor(GapRoutedGNNPreprocessor):
    """Same as GapRoutedGNNPreprocessor but drops samples with gap >= threshold
    (fine-tune stage 2, routed sub-distribution only)."""

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


@register_preprocessor("syndrome_gnn")
class SyndromeGNNPreprocessor(GapRoutedGNNPreprocessor):
    """Syndrome-only GNN preprocessor for ablation.

    Same graph topology + static edge weight as GapRoutedGNN, but drops the
    E_0/E_1 binary indicators (d_edge=1: just [w_e_norm]). Matches Syndrome3D
    role for the CNN ablation pair.
    """

    def __init__(self, circuit, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.d_edge = 1
        self.n_scalars = 0

    def get_model_kwargs(self) -> Dict[str, Any]:
        kw = super().get_model_kwargs()
        kw["d_edge"] = 1
        kw["n_scalars"] = 0
        return kw

    def cpu_transform(self, batch):
        out = super().cpu_transform(batch)
        # syndrome-only ablation: drop E_0/E_1 indicators + scalars
        return {"ch_syn": out["ch_syn"]}

    def gpu_transform(self, batch_data):
        ch_syn = batch_data["ch_syn"]
        B = ch_syn.size(0)
        device = ch_syn.device

        node_x = torch.zeros(B, self.n_node, device=device, dtype=torch.float32)
        node_x[:, : self.n_det] = ch_syn.float()

        edge_w_norm = self._edge_w_norm.to(device)                  # (n_undir,)
        edge_feat = edge_w_norm.view(1, -1, 1).expand(B, -1, 1)     # (B, n_undir, 1)

        flat = torch.cat([
            node_x,
            edge_feat.reshape(B, -1),
        ], dim=-1)
        return flat


@register_preprocessor("syndrome_gnn_filtered")
class SyndromeGNNFilteredPreprocessor(SyndromeGNNPreprocessor):
    """Syndrome-only GNN preprocessor + gap-based filtering (stage 2 ablation)."""

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, gap_threshold: float = 2.0, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.gap_threshold = float(gap_threshold)

    def cpu_transform(self, batch):
        full = GapRoutedGNNPreprocessor.cpu_transform(self, batch)
        gaps = full["scalars"][:, 0]
        return {
            "ch_syn":     full["ch_syn"],
            "_keep_mask": (gaps < self.gap_threshold),
        }


# ─────────────────────────────────────────────────────────────────────────
# Cached variants — read pre-computed MWPM byproducts from offline .npz
# (skip the per-sample MWPM aug 2× cost at training time).
#
# The .npz is produced by scripts/build_routed_cache.py and is *shared*
# across CNN/GNN families (see also cached_gap_routed_cnn, cached_syndrome_3d).
#
# .npz layout:
#   syndromes   (N, n_det)         uint8
#   observables (N, 1)              uint8
#   edge_in_E0  (N, n_undir_e)     uint8
#   edge_in_E1  (N, n_undir_e)     uint8
#   scalars     (N, 6)             float32
# ─────────────────────────────────────────────────────────────────────────


@register_preprocessor("cached_gap_routed_gnn")
class CachedGapRoutedGNNPreprocessor(GapRoutedGNNPreprocessor):
    """Offline-cached GapRoutedGNN preprocessor.

    Reads pre-computed MWPM-aug byproducts straight from .npz — no per-sample
    MWPM call. The .npz should already contain only the filtered (gap < τ)
    sub-distribution; no further filtering applied here.
    """

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes", "edge_in_E0", "edge_in_E1", "scalars"]

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # QECRawDataset path → per-sample (1D) tensors.
        # OnlineQECDataset path is not used for cached preprocessors.
        return {
            "ch_syn":     sample["syndromes"].float(),
            "edge_in_E0": sample["edge_in_E0"].float(),
            "edge_in_E1": sample["edge_in_E1"].float(),
            "scalars":    sample["scalars"].float(),
        }


@register_preprocessor("cached_syndrome_gnn")
class CachedSyndromeGNNPreprocessor(SyndromeGNNPreprocessor):
    """Cached syndrome-only GNN ablation — reads only `syndromes` from the
    same shared .npz produced by scripts/build_routed_cache.py."""

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes"]

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"ch_syn": sample["syndromes"].float()}


# ─────────────────────────────────────────────────────────────────────────
# CachedGapRoutedGNNCNNFeatPreprocessor — decoding-graph topology + CNN-style
# per-detector aggregated features. Tests the information-bottleneck
# hypothesis: same topology as bipartite trap, but edge-level MWPM info is
# *aggregated onto detectors* (CNN paper's 4-channel projection) → lossy.
#
# Node features (4-dim per detector):
#   [syn, diff, w0_sum, w1_sum]   (same per-detector channels as CNN)
# Edge features (1-dim per fault edge):
#   [w_e_norm]                    (static prior only; per-shot MWPM info
#                                  no longer survives at the edge level)
# Scalars: REMOVED                 (block mwpm_pred shortcut to head)
#
# Output layout:
#   [node_x_flat (B, n_node * 4)] + [edge_feat (B, n_undir_e * 1)]
# ─────────────────────────────────────────────────────────────────────────


@register_preprocessor("cached_gap_routed_gnn_cnnfeat")
class CachedGapRoutedGNNCNNFeatPreprocessor(CachedGapRoutedGNNPreprocessor):
    """CNN-style aggregated features on decoding-graph topology (mono GNN)."""

    def __init__(self, circuit, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        import numpy as np
        # Override feature dims for the model
        self.d_node_in = 4
        self.d_edge = 1
        self.n_scalars = 0

        # Build per-undirected-edge endpoint + raw weight buffers for the
        # per-detector aggregation (mirrors CachedGapRoutedCNNPreprocessor).
        ei = self._edge_index_np            # (2, 2*n_undir_e)
        endpoints_a = ei[0, ::2].astype(np.int64)   # (n_undir_e,)
        endpoints_b = ei[1, ::2].astype(np.int64)

        # Recover per-undir-edge w_static (denormalize from w_e_norm is fragile;
        # just rebuild from the EC indices stored implicitly via undir_to_efidx
        # which we recompute here from the topology constructed in the parent).
        from beliefmatching import detector_error_model_to_check_matrices
        from scipy import sparse as sp
        dem = circuit.detector_error_model(decompose_errors=True)
        mtx = detector_error_model_to_check_matrices(dem)
        priors = np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors)
        w_static_full = -np.log(np.clip(priors, 1e-14, 1 - 1e-14)).astype(np.float32)
        EC = sp.csc_matrix(mtx.edge_check_matrix).tocsc()
        n_edges_total = EC.shape[1]
        VIRTUAL = self.n_det
        edge_w_list = []
        for e_idx in range(n_edges_total):
            rows = EC.getcol(e_idx).nonzero()[0]
            if rows.size == 1 or rows.size == 2:
                edge_w_list.append(w_static_full[e_idx])
        edge_w_raw = np.asarray(edge_w_list, dtype=np.float32)
        assert edge_w_raw.shape[0] == self.n_undir_e, (
            f"undir edge count mismatch: {edge_w_raw.shape[0]} vs {self.n_undir_e}"
        )

        self.register_buffer(
            "_edge_endpoint_a",
            torch.from_numpy(endpoints_a).long(),
            persistent=False,
        )
        self.register_buffer(
            "_edge_endpoint_b",
            torch.from_numpy(endpoints_b).long(),
            persistent=False,
        )
        self.register_buffer(
            "_edge_w_raw",
            torch.from_numpy(edge_w_raw),
            persistent=False,
        )

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes", "edge_in_E0", "edge_in_E1"]

    def get_model_kwargs(self) -> Dict[str, Any]:
        kw = super().get_model_kwargs()
        kw["d_node_in"] = self.d_node_in
        kw["d_edge"] = self.d_edge
        kw["n_scalars"] = self.n_scalars
        return kw

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Pass through; aggregation runs on GPU in gpu_transform.
        return {
            "ch_syn":     sample["syndromes"].float(),
            "edge_in_E0": sample["edge_in_E0"].float(),
            "edge_in_E1": sample["edge_in_E1"].float(),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ch_syn = batch_data["ch_syn"]
        e0 = batch_data["edge_in_E0"]
        e1 = batch_data["edge_in_E1"]
        B = ch_syn.size(0)
        device = ch_syn.device
        n_det = self.n_det

        a = self._edge_endpoint_a.to(device)            # (n_undir_e,)
        b = self._edge_endpoint_b.to(device)
        w = self._edge_w_raw.to(device)                 # (n_undir_e,)

        # Per-detector aggregation (only edges incident to a real detector;
        # virtual boundary endpoint = n_det → masked out).
        valid_a = a < n_det
        valid_b = b < n_det
        a_valid = a[valid_a]
        b_valid = b[valid_b]
        a_idx = a_valid.view(1, -1).expand(B, -1)
        b_idx = b_valid.view(1, -1).expand(B, -1)

        ch_w0 = torch.zeros(B, n_det, device=device, dtype=torch.float32)
        ch_w1 = torch.zeros(B, n_det, device=device, dtype=torch.float32)
        ch_diff = torch.zeros(B, n_det, device=device, dtype=torch.float32)

        contrib_e0 = e0 * w.view(1, -1)
        contrib_e1 = e1 * w.view(1, -1)
        ch_w0.scatter_add_(1, a_idx, contrib_e0[:, valid_a])
        ch_w0.scatter_add_(1, b_idx, contrib_e0[:, valid_b])
        ch_w1.scatter_add_(1, a_idx, contrib_e1[:, valid_a])
        ch_w1.scatter_add_(1, b_idx, contrib_e1[:, valid_b])

        diff = (e0.bool() ^ e1.bool()).float()
        ch_diff.scatter_reduce_(1, a_idx, diff[:, valid_a], reduce="amax",
                                 include_self=True)
        ch_diff.scatter_reduce_(1, b_idx, diff[:, valid_b], reduce="amax",
                                 include_self=True)

        # node_x: (B, n_node, 4) — virtual boundary node gets all zeros.
        node_x = torch.zeros(B, self.n_node, 4, device=device, dtype=torch.float32)
        node_x[:, :n_det, 0] = ch_syn.float()
        node_x[:, :n_det, 1] = ch_diff
        node_x[:, :n_det, 2] = ch_w0
        node_x[:, :n_det, 3] = ch_w1

        # edge feature: static w_e_norm only (1-dim)
        edge_w_norm = self._edge_w_norm.to(device)              # (n_undir_e,)
        edge_feat = edge_w_norm.view(1, -1, 1).expand(B, -1, 1) # (B, n_undir_e, 1)

        flat = torch.cat([
            node_x.reshape(B, -1),
            edge_feat.reshape(B, -1),
        ], dim=-1)
        return flat
