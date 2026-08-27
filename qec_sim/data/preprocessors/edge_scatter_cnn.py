"""Edge-scatter CNN preprocessor.

Combines GNN's edge-endpoint topology with CNN's (T, H, W) grid layout.
The model receives per-edge features (B, n_undir_e, d_edge) and scatters them
onto a detector-indexed (T, H, W) grid inside the forward pass.

Default output layout (use_w_norm=False):
    edge_feat_flat: shape (B, n_undir_e * 2) = [1[E_0], 1[E_1]] per edge

Legacy layout (use_w_norm=True): compat for old checkpoints that were trained
with the static edge-weight feature. Adds w_e_norm as the first per-edge slot:
    edge_feat_flat: shape (B, n_undir_e * 3) = [w_e_norm, 1[E_0], 1[E_1]] per edge

This pairs with edge_scatter_cnn family models (F1/F2/F3/A).
"""
import torch
import numpy as np
from typing import Dict, Any, List
from qec_sim.data.registry import register_preprocessor
from qec_sim.data.preprocessors.gap_routed_cnn import CachedGapRoutedCNNPreprocessor


@register_preprocessor("cached_edge_scatter_cnn")
class CachedEdgeScatterCNNPreprocessor(CachedGapRoutedCNNPreprocessor):
    """Edge-scatter CNN preprocessor (cached variant).

    Reads from the shared routed cache (same .npz as cached_gap_routed_cnn /
    cached_gap_routed_gnn).
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, use_w_norm: bool = False, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.use_w_norm = bool(use_w_norm)
        # Parent (CachedGapRoutedCNN) gave us T, H, W, _flat_idx (det idx → grid).
        # Now build the edge-endpoint topology (mirrors CachedGapRoutedGNN).
        from scipy import sparse as sp
        from beliefmatching import detector_error_model_to_check_matrices

        dem = circuit.detector_error_model(decompose_errors=True)
        mtx = detector_error_model_to_check_matrices(dem)
        EC = sp.csc_matrix(mtx.edge_check_matrix).tocsc()

        n_edges_total = EC.shape[1]
        VIRTUAL = self.num_detectors

        endpoints_a, endpoints_b, edge_w_list = [], [], []
        if self.use_w_norm:
            priors = np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors)
            w_static = -np.log(np.clip(priors, 1e-14, 1 - 1e-14)).astype(np.float32)

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
            if self.use_w_norm:
                edge_w_list.append(w_static[e_idx])

        self.n_undir_e = len(endpoints_a)

        # numpy arrays for the model kwargs
        self._edge_endpoint_a_np = np.asarray(endpoints_a, dtype=np.int64)
        self._edge_endpoint_b_np = np.asarray(endpoints_b, dtype=np.int64)

        if self.use_w_norm:
            edge_w_raw = np.asarray(edge_w_list, dtype=np.float32)
            w_mean = float(edge_w_raw.mean())
            w_std = float(edge_w_raw.std() + 1e-9)
            edge_w_norm = ((edge_w_raw - w_mean) / w_std).astype(np.float32)
            self.register_buffer(
                "_edge_w_norm",
                torch.from_numpy(edge_w_norm),
                persistent=False,
            )

    @property
    def required_data_keys(self) -> List[str]:
        return ["syndromes", "edge_in_E0", "edge_in_E1"]

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {
            "T": self.T, "H": self.H, "W": self.W,
            "n_det": self.num_detectors,
            "n_undir_e": self.n_undir_e,
            "d_edge": 3 if self.use_w_norm else 2,
            "edge_endpoint_a_np": self._edge_endpoint_a_np,
            "edge_endpoint_b_np": self._edge_endpoint_b_np,
            "det_flat_idx_np": self._flat_idx.cpu().numpy().astype(np.int64),
            "n_scalars": 0,
        }

    def cpu_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "edge_in_E0": sample["edge_in_E0"].float(),
            "edge_in_E1": sample["edge_in_E1"].float(),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        e0 = batch_data["edge_in_E0"]
        e1 = batch_data["edge_in_E1"]
        B = e0.size(0)

        if self.use_w_norm:
            edge_w_b = self._edge_w_norm.to(e0.device).unsqueeze(0).expand(B, -1)
            edge_feat = torch.stack([edge_w_b, e0, e1], dim=-1)   # (B, n_undir_e, 3)
        else:
            edge_feat = torch.stack([e0, e1], dim=-1)              # (B, n_undir_e, 2)
        return edge_feat.reshape(B, -1)
