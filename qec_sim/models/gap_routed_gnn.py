"""GapRoutedGNN — message-passing GNN that consumes the complementary-gap
byproducts as **edge features** (rather than CNN's per-detector grid projection).

Designed as a drop-in replacement for the CNN strong stage of the gap-routed
confidence router. Trained on the full data distribution; at inference time the
router calls this model only when gap < τ (paired with GapRoutedGNN3D decoder).

Static graph topology (fixed per circuit, injected via get_model_kwargs):
  n_node              num_detectors + 1 (virtual boundary node)
  edge_index_np       (2, 2*n_undir_e) bidirected
  edge_w_norm_np      (n_undir_e,)     standardized -log p_e
  undir_idx_for_dir   (2*n_undir_e,)   maps directed slot → undirected edge

Per-shot inputs (unpacked from the flat tensor produced by GapRoutedGNNPreprocessor):
  node_x[B, n_node]                       (syndrome on detectors, 0 on virtual)
  edge_feat_undir[B, n_undir_e, d_edge]   = [w_e_norm, 1[E_0], 1[E_1]] (d_edge=3)
  scalars[B, n_scalars]                   (gap, w_0, w_1, weight_sum, n_matched_norm, mwpm_pred)

Forward:
  h ← Linear(node_x.unsqueeze(-1))                # (B, n_node, d_hidden)
  for L layers:
      m_{u→v} = MLP_msg([h_u, h_v, edge_feat_dir[u,v]])
      agg_v   = Σ_{u} m_{u→v}                     # scatter_add to dst
      h ← h + LayerNorm(MLP_upd([h, agg]))         # residual
  g = mean_v h                                     # graph pooling
  return MLP_head([g, scalars])
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


class EdgeConcatMPNNLayer(nn.Module):
    """One message-passing step over fixed bidirected topology.

    Operates on (B, n_node, d_h) node features and (B, n_edge_dir, d_e) directed
    edge features. src/dst are 1D LongTensors of shape (n_edge_dir,), shared
    across the batch (fixed topology).
    """

    def __init__(self, d_h: int, d_e: int, msg_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * d_h + d_e, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(msg_hidden, d_h),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * d_h, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, d_h),
        )
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: torch.Tensor, edge_feat_dir: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        B, n_node, d_h = h.shape
        E = src.shape[0]

        h_src = h.index_select(1, src)                  # (B, E, d_h)
        h_dst = h.index_select(1, dst)                  # (B, E, d_h)
        m = self.msg(torch.cat([h_src, h_dst, edge_feat_dir], dim=-1))  # (B, E, d_h)

        # scatter_add over destination
        agg = torch.zeros(B, n_node, d_h, device=h.device, dtype=h.dtype)
        dst_idx = dst.view(1, E, 1).expand(B, -1, d_h)
        agg.scatter_add_(1, dst_idx, m)

        delta = self.upd(torch.cat([h, agg], dim=-1))
        return h + self.norm(delta)


@register_model("gap_routed_gnn")
class GapRoutedGNN(BaseQECModel):
    REQUIRED_PREPROCESSOR = "gap_routed_gnn"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_edge: int = 3,
        n_scalars: int = 6,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        d_hidden: int = 64,
        n_layers: int = 3,
        msg_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.2,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or edge_w_norm_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "GapRoutedGNN requires edge_index_np / edge_w_norm_np / "
                "undir_idx_for_dir_np (injected by the preprocessor)."
            )

        self.n_node = n_node
        self.n_undir_e = n_undir_e
        self.d_edge = d_edge
        self.n_scalars = n_scalars
        self.d_hidden = d_hidden
        self.pool = pool

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        # edge_w_norm is also delivered by the preprocessor on every batch, so we
        # don't strictly need it here. We keep it as a buffer for inference-time
        # use (decoder builds the model directly without going through the
        # preprocessor's gpu_transform).
        edge_w_norm = torch.as_tensor(edge_w_norm_np, dtype=torch.float32)
        self.register_buffer("_src", edge_index[0].contiguous(), persistent=False)
        self.register_buffer("_dst", edge_index[1].contiguous(), persistent=False)
        self.register_buffer("_undir_idx_for_dir", undir_idx_for_dir.contiguous(),
                             persistent=False)
        self.register_buffer("_edge_w_norm", edge_w_norm.contiguous(), persistent=False)

        self.node_emb = nn.Linear(1, d_hidden)
        self.layers = nn.ModuleList([
            EdgeConcatMPNNLayer(d_hidden, d_edge, msg_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_hidden + n_scalars, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_node = self.n_node
        n_undir = self.n_undir_e
        d_e = self.d_edge

        edge_block = n_undir * d_e
        node_x = x[:, :n_node].unsqueeze(-1)                                    # (B, n_node, 1)
        edge_undir = x[:, n_node : n_node + edge_block].view(B, n_undir, d_e)   # (B, n_undir, d_e)
        if self.n_scalars > 0:
            scalars = x[:, n_node + edge_block : n_node + edge_block + self.n_scalars]
        else:
            scalars = x.new_zeros(B, 0)

        # bidirected edge features
        edge_dir = edge_undir.index_select(1, self._undir_idx_for_dir)          # (B, 2*n_undir, d_e)

        h = self.node_emb(node_x)                                               # (B, n_node, d_h)
        for layer in self.layers:
            h = layer(h, edge_dir, self._src, self._dst)

        if self.pool == "mean":
            g = h.mean(dim=1)
        elif self.pool == "sum":
            g = h.sum(dim=1)
        elif self.pool == "max":
            g = h.max(dim=1).values
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        return self.head(torch.cat([g, scalars], dim=-1) if self.n_scalars > 0 else g)


@register_model("gap_routed_gnn_filtered")
class GapRoutedGNNFiltered(GapRoutedGNN):
    """Same as GapRoutedGNN but trained on the gap-filtered sub-distribution
    via GapRoutedGNNFilteredPreprocessor (Stage 2 fine-tune)."""
    REQUIRED_PREPROCESSOR = "gap_routed_gnn_filtered"


@register_model("syndrome_gnn")
class SyndromeGNN(GapRoutedGNN):
    """Syndrome-only GNN baseline for ablation (no E_0/E_1 indicators).

    Same architecture as GapRoutedGNN but consumes only the syndrome + static
    edge weight (d_edge=1, n_scalars=0). Pairs with SyndromeGNNPreprocessor.
    """
    REQUIRED_PREPROCESSOR = "syndrome_gnn"

    def __init__(self, num_observables: int, n_node: int, n_undir_e: int,
                 d_edge: int = 1, n_scalars: int = 0, **kwargs):
        super().__init__(
            num_observables=num_observables,
            n_node=n_node, n_undir_e=n_undir_e,
            d_edge=d_edge, n_scalars=n_scalars,
            **kwargs,
        )


@register_model("syndrome_gnn_filtered")
class SyndromeGNNFiltered(SyndromeGNN):
    """Syndrome-only GNN with gap-filtered fine-tuning preprocessor."""
    REQUIRED_PREPROCESSOR = "syndrome_gnn_filtered"


# ─────────────────────────────────────────────────────────────────────────
# Cached pairs — identical architecture, point to cached preprocessor that
# reads from offline .npz (no per-sample MWPM at training time).
# ─────────────────────────────────────────────────────────────────────────

@register_model("cached_gap_routed_gnn")
class CachedGapRoutedGNN(GapRoutedGNN):
    """Same architecture as GapRoutedGNN; consumes cached preprocessor."""
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn"


@register_model("cached_syndrome_gnn")
class CachedSyndromeGNN(SyndromeGNN):
    """Same architecture as SyndromeGNN; consumes cached syndrome preprocessor."""
    REQUIRED_PREPROCESSOR = "cached_syndrome_gnn"


# ─────────────────────────────────────────────────────────────────────────
# Bipartite variant — Tanner-style factor graph view of the same data.
# Variable nodes = fault edges (one per decoding-graph edge),
# check nodes    = detectors. Each variable connects to its 2 endpoint
# detectors via bipartite incidence edges (2 per fault edge → 2 × n_undir_e
# incidence pairs total).
#
# Same input flat tensor as the mono GapRoutedGNN: byproducts (w_e_norm,
# 1[E_0], 1[E_1]) now live as variable-node features instead of edge features.
# ─────────────────────────────────────────────────────────────────────────


class BipartiteMPNNLayer(nn.Module):
    """One bipartite round = variable → check pass + check → variable pass.

    Both phases use the same incidence (var_inc, check_inc) but in opposite
    direction. Each phase is scatter_add aggregation + residual MLP + LN.
    """

    def __init__(self, d_h: int, msg_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.msg_v2c = nn.Sequential(
            nn.Linear(d_h, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(msg_hidden, d_h),
        )
        self.upd_c = nn.Sequential(
            nn.Linear(2 * d_h, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, d_h),
        )
        self.norm_c = nn.LayerNorm(d_h)

        self.msg_c2v = nn.Sequential(
            nn.Linear(d_h, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(msg_hidden, d_h),
        )
        self.upd_v = nn.Sequential(
            nn.Linear(2 * d_h, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, d_h),
        )
        self.norm_v = nn.LayerNorm(d_h)

    def forward(self, h_var: torch.Tensor, h_check: torch.Tensor,
                var_inc: torch.Tensor, check_inc: torch.Tensor) -> tuple:
        """
        h_var:    (B, n_var, d_h)
        h_check:  (B, n_check, d_h)
        var_inc:  (E_inc,) variable index for each bipartite incidence
        check_inc:(E_inc,) check index for each bipartite incidence
        """
        B, n_var, d_h = h_var.shape
        n_check = h_check.shape[1]
        E = var_inc.shape[0]

        # Phase A: variable → check
        h_src_v = h_var.index_select(1, var_inc)            # (B, E, d_h)
        m_v2c = self.msg_v2c(h_src_v)                       # (B, E, d_h)
        agg_c = torch.zeros(B, n_check, d_h, device=h_var.device, dtype=h_var.dtype)
        dst_idx = check_inc.view(1, E, 1).expand(B, -1, d_h)
        agg_c.scatter_add_(1, dst_idx, m_v2c)
        delta_c = self.upd_c(torch.cat([h_check, agg_c], dim=-1))
        h_check = h_check + self.norm_c(delta_c)

        # Phase B: check → variable
        h_src_c = h_check.index_select(1, check_inc)        # (B, E, d_h)
        m_c2v = self.msg_c2v(h_src_c)                       # (B, E, d_h)
        agg_v = torch.zeros(B, n_var, d_h, device=h_var.device, dtype=h_var.dtype)
        src_idx = var_inc.view(1, E, 1).expand(B, -1, d_h)
        agg_v.scatter_add_(1, src_idx, m_c2v)
        delta_v = self.upd_v(torch.cat([h_var, agg_v], dim=-1))
        h_var = h_var + self.norm_v(delta_v)

        return h_var, h_check


@register_model("gap_routed_gnn_bipartite")
class GapRoutedGNNBipartite(BaseQECModel):
    """Tanner-style bipartite factor-graph GNN over the decoding graph.

    Consumes the same input layout as the mono GapRoutedGNN preprocessor:
        [node_x (n_node)] + [edge_feat (n_undir_e * d_edge)] + [scalars]
    Re-interprets:
        - check nodes  = original n_node (detectors + virtual boundary)
        - variable nodes = n_undir_e fault edges
        - variable features = the d_edge per-edge byproducts (w_e_norm, E_0, E_1)
        - check features    = the per-detector syndrome bit

    Bipartite incidence is derived from the static `edge_index_np`:
        for each undir edge i, both directed slots (a,b) and (b,a) yield two
        (variable=i, check=endpoint) incidence pairs.
    """

    REQUIRED_PREPROCESSOR = "gap_routed_gnn"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_edge: int = 3,
        n_scalars: int = 6,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        d_hidden: int = 64,
        n_layers: int = 3,
        msg_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.2,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "GapRoutedGNNBipartite requires edge_index_np and "
                "undir_idx_for_dir_np (injected by the preprocessor)."
            )

        self.n_check = n_node
        self.n_var = n_undir_e
        self.d_var_in = d_edge
        self.n_scalars = n_scalars
        self.d_hidden = d_hidden
        self.pool = pool

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        # Bipartite incidence: each directed slot of the mono edge_index is one
        # (variable=undir_idx, check=src) incidence row. Since each undir edge
        # appears in both directions, this yields exactly 2 incidence rows per
        # variable — one to each of its two endpoint detectors.
        var_inc = undir_idx_for_dir
        check_inc = edge_index[0]

        self.register_buffer("_var_inc", var_inc.contiguous(), persistent=False)
        self.register_buffer("_check_inc", check_inc.contiguous(), persistent=False)

        self.check_emb = nn.Linear(1, d_hidden)
        self.var_emb = nn.Linear(d_edge, d_hidden)

        self.layers = nn.ModuleList([
            BipartiteMPNNLayer(d_hidden, msg_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Pool over the two node types separately, concat, then head.
        head_in = 2 * d_hidden + n_scalars
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_check = self.n_check
        n_var = self.n_var
        d_e = self.d_var_in
        edge_block = n_var * d_e

        check_x = x[:, :n_check].unsqueeze(-1)                              # (B, n_check, 1)
        var_x = x[:, n_check : n_check + edge_block].view(B, n_var, d_e)    # (B, n_var, d_e)
        if self.n_scalars > 0:
            scalars = x[:, n_check + edge_block : n_check + edge_block + self.n_scalars]
        else:
            scalars = x.new_zeros(B, 0)

        h_check = self.check_emb(check_x)
        h_var = self.var_emb(var_x)

        for layer in self.layers:
            h_var, h_check = layer(h_var, h_check, self._var_inc, self._check_inc)

        if self.pool == "mean":
            g_check = h_check.mean(dim=1)
            g_var = h_var.mean(dim=1)
        elif self.pool == "sum":
            g_check = h_check.sum(dim=1)
            g_var = h_var.sum(dim=1)
        elif self.pool == "max":
            g_check = h_check.max(dim=1).values
            g_var = h_var.max(dim=1).values
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        parts = [g_check, g_var]
        if self.n_scalars > 0:
            parts.append(scalars)
        return self.head(torch.cat(parts, dim=-1))


@register_model("cached_gap_routed_gnn_bipartite")
class CachedGapRoutedGNNBipartite(GapRoutedGNNBipartite):
    """Bipartite GNN consuming the cached shared .npz."""
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn"


# ─────────────────────────────────────────────────────────────────────────
# ASTRA-style variant — bipartite Tanner-form + GRU recurrent update.
#
# Adapted from Arshpreet Maan's ASTRA QLDPC GNN decoder
# (github.com/arshpreetmaan/astra):
#   - Bipartite graph: check nodes (detectors) + variable nodes (fault edges).
#   - GRU update with initial input fed back at each iteration (keeps the raw
#     syndrome / byproducts signal alive across iterations, helps avoid
#     over-smoothing).
#   - Many iterations (L=5+) with small hidden dim (h=16+) → tiny param budget.
#   - No mwpm_pred shortcut: scalars NOT injected into the head (forces the
#     model to learn from the bipartite message passing alone).
#
# Input layout still matches GapRoutedGNNPreprocessor:
#   x[:, :n_node]                             check node syndrome (+ virtual)
#   x[:, n_node : n_node + n_undir_e * 3]    var node features (byproducts)
#   x[:, last 6]                              scalars (ignored by default)
# ─────────────────────────────────────────────────────────────────────────


@register_model("gap_routed_gnn_astra_style")
class GapRoutedGNNAstraStyle(BaseQECModel):
    """ASTRA-style bipartite GRU MPNN for surface-code decoding.

    Key differences from `GapRoutedGNNBipartite`:
      - GRU update instead of MLP+residual+LayerNorm
      - Initial input (syndrome/byproducts) re-fed at every iteration via
        the GRU input gate
      - Defaults to much smaller hidden dim and more iterations
      - Scalars NOT used by default (avoids mwpm_pred shortcut trap)
    """
    REQUIRED_PREPROCESSOR = "gap_routed_gnn"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_edge: int = 3,
        n_scalars: int = 6,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        h_dim: int = 16,
        n_iters: int = 5,
        msg_hidden: int = 64,
        head_hidden: int = 32,
        dropout: float = 0.1,
        use_scalars_in_head: bool = False,  # set True to allow head to see gap/w_0/.../mwpm_pred
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "GapRoutedGNNAstraStyle requires edge_index_np and undir_idx_for_dir_np"
            )
        self.n_check = n_node
        self.n_var = n_undir_e
        self.d_var_in = d_edge
        self.n_scalars = n_scalars
        self.h_dim = h_dim
        self.n_iters = n_iters
        self.use_scalars_in_head = use_scalars_in_head

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        # Bipartite incidence: each directed slot maps to (variable, check) pair.
        self.register_buffer("_var_inc", undir_idx_for_dir.contiguous(), persistent=False)
        self.register_buffer("_check_inc", edge_index[0].contiguous(), persistent=False)

        # Initial embeddings — small, just lift to h_dim
        self.check_emb = nn.Linear(1, h_dim)
        self.var_emb = nn.Linear(d_edge, h_dim)

        # Message networks (ASTRA-style: 3-layer MLP with dropout)
        edge_feat_dim = h_dim
        self.msg_v2c = nn.Sequential(
            nn.Linear(2 * h_dim, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, edge_feat_dim),
        )
        self.msg_c2v = nn.Sequential(
            nn.Linear(2 * h_dim, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, edge_feat_dim),
        )

        # GRU update (ASTRA core: recurrent across iterations)
        # GRU input at each iter: [aggregated message, raw initial input]
        # GRU state: current node hidden (h_dim)
        self.gru_c = nn.GRUCell(edge_feat_dim + 1, h_dim)        # +1 for syndrome bit
        self.gru_v = nn.GRUCell(edge_feat_dim + d_edge, h_dim)   # +d_edge for byproducts

        # Output head — graph pool of check + var, optional scalars
        pool_dim = 2 * h_dim  # concat(check_pool, var_pool)
        extra = n_scalars if use_scalars_in_head else 0
        self.head = nn.Sequential(
            nn.Linear(pool_dim + extra, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_check = self.n_check
        n_var = self.n_var
        d_e = self.d_var_in
        h = self.h_dim
        edge_block = n_var * d_e

        check_init = x[:, :n_check].unsqueeze(-1)                              # (B, n_check, 1)
        var_init = x[:, n_check : n_check + edge_block].view(B, n_var, d_e)    # (B, n_var, d_e)
        if self.n_scalars > 0 and self.use_scalars_in_head:
            scalars = x[:, n_check + edge_block : n_check + edge_block + self.n_scalars]
        else:
            scalars = x.new_zeros(B, 0)

        h_c = self.check_emb(check_init)        # (B, n_check, h)
        h_v = self.var_emb(var_init)            # (B, n_var, h)

        var_inc = self._var_inc                 # (E,)
        check_inc = self._check_inc             # (E,)
        E = var_inc.shape[0]

        for _ in range(self.n_iters):
            # Phase A: variable → check
            h_src_v = h_v.index_select(1, var_inc)             # (B, E, h)
            h_dst_c = h_c.index_select(1, check_inc)
            m_v2c = self.msg_v2c(torch.cat([h_src_v, h_dst_c], dim=-1))  # (B, E, h)
            agg_c = torch.zeros(B, n_check, h, device=m_v2c.device, dtype=m_v2c.dtype)
            agg_c.scatter_add_(1, check_inc.view(1, E, 1).expand(B, -1, h), m_v2c)
            # GRU: input = [agg_c, raw syndrome], state = h_c
            gru_in_c = torch.cat([agg_c, check_init], dim=-1)              # (B, n_check, h+1)
            h_c = self.gru_c(gru_in_c.reshape(-1, h + 1),
                             h_c.reshape(-1, h)).view(B, n_check, h)

            # Phase B: check → variable
            h_src_c = h_c.index_select(1, check_inc)
            h_dst_v = h_v.index_select(1, var_inc)
            m_c2v = self.msg_c2v(torch.cat([h_src_c, h_dst_v], dim=-1))    # (B, E, h)
            agg_v = torch.zeros(B, n_var, h, device=m_c2v.device, dtype=m_c2v.dtype)
            agg_v.scatter_add_(1, var_inc.view(1, E, 1).expand(B, -1, h), m_c2v)
            gru_in_v = torch.cat([agg_v, var_init], dim=-1)                # (B, n_var, h+d_e)
            h_v = self.gru_v(gru_in_v.reshape(-1, h + d_e),
                             h_v.reshape(-1, h)).view(B, n_var, h)

        # Pool both node types, concat, head
        g_check = h_c.mean(dim=1)               # (B, h)
        g_var = h_v.mean(dim=1)
        parts = [g_check, g_var]
        if self.n_scalars > 0 and self.use_scalars_in_head:
            parts.append(scalars)
        return self.head(torch.cat(parts, dim=-1))


@register_model("cached_gap_routed_gnn_astra_style")
class CachedGapRoutedGNNAstraStyle(GapRoutedGNNAstraStyle):
    """ASTRA-style GNN consuming the cached shared .npz."""
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn"


@register_model("cached_gap_routed_gnn_bipartite_cnnfeat")
class CachedGapRoutedGNNBipartiteCNNFeat(BaseQECModel):
    """Bipartite variant of mono CNN-feat: same input layout (4-dim per
    detector + 1-dim per fault edge), but uses var nodes + check nodes
    with bipartite message passing.

    Information content identical to mono cnnfeat (per-shot info only in
    check_x; var_x is static w_e_norm). Tests whether bipartite topology
    itself adds value when both node types are info-restricted.

    Input layout (from CachedGapRoutedGNNCNNFeatPreprocessor):
      [check_x_flat (B, n_node * 4)]     — 4 CNN-style channels per detector
      [var_x_flat (B, n_undir_e * 1)]    — w_e_norm only (static)
      (no scalars)
    """
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn_cnnfeat"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_node_in: int = 4,
        d_edge: int = 1,
        n_scalars: int = 0,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        d_hidden: int = 64,
        n_layers: int = 3,
        msg_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.2,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "CachedGapRoutedGNNBipartiteCNNFeat requires edge_index_np / "
                "undir_idx_for_dir_np from the preprocessor."
            )

        self.n_check = n_node
        self.n_var = n_undir_e
        self.d_node_in = d_node_in
        self.d_var_in = d_edge
        self.d_hidden = d_hidden
        self.pool = pool

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        var_inc = undir_idx_for_dir
        check_inc = edge_index[0]
        self.register_buffer("_var_inc", var_inc.contiguous(), persistent=False)
        self.register_buffer("_check_inc", check_inc.contiguous(), persistent=False)

        self.check_emb = nn.Linear(d_node_in, d_hidden)
        self.var_emb = nn.Linear(d_edge, d_hidden)

        self.layers = nn.ModuleList([
            BipartiteMPNNLayer(d_hidden, msg_hidden, dropout)
            for _ in range(n_layers)
        ])

        head_in = 2 * d_hidden  # no scalars
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_check = self.n_check
        n_var = self.n_var
        d_in = self.d_node_in
        d_e = self.d_var_in

        node_block = n_check * d_in
        edge_block = n_var * d_e

        check_x = x[:, :node_block].view(B, n_check, d_in)
        var_x = x[:, node_block : node_block + edge_block].view(B, n_var, d_e)

        h_check = self.check_emb(check_x)
        h_var = self.var_emb(var_x)

        for layer in self.layers:
            h_var, h_check = layer(h_var, h_check, self._var_inc, self._check_inc)

        if self.pool == "mean":
            g_check = h_check.mean(dim=1)
            g_var = h_var.mean(dim=1)
        elif self.pool == "sum":
            g_check = h_check.sum(dim=1)
            g_var = h_var.sum(dim=1)
        elif self.pool == "max":
            g_check = h_check.max(dim=1).values
            g_var = h_var.max(dim=1).values
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        return self.head(torch.cat([g_check, g_var], dim=-1))


@register_model("cached_gap_routed_gnn_astra_style_cnnfeat")
class CachedGapRoutedGNNAstraStyleCNNFeat(BaseQECModel):
    """ASTRA-style variant of mono CNN-feat: bipartite GRU MPNN that
    consumes the same input layout as the mono/bipartite cnnfeat models
    (4-dim per detector check_x + 1-dim per fault edge var_x).

    Combines ASTRA's iterative-GRU strength with CNN-feat's information
    bottleneck. Tests whether ASTRA's expressivity is enough to escape
    the trap when feature aggregation is also applied.

    Input layout (same as bipartite_cnnfeat).
    """
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn_cnnfeat"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_node_in: int = 4,
        d_edge: int = 1,
        n_scalars: int = 0,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        h_dim: int = 16,
        n_iters: int = 5,
        msg_hidden: int = 64,
        head_hidden: int = 32,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "CachedGapRoutedGNNAstraStyleCNNFeat requires edge_index_np / "
                "undir_idx_for_dir_np from the preprocessor."
            )
        self.n_check = n_node
        self.n_var = n_undir_e
        self.d_node_in = d_node_in
        self.d_var_in = d_edge
        self.h_dim = h_dim
        self.n_iters = n_iters

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        self.register_buffer("_var_inc", undir_idx_for_dir.contiguous(), persistent=False)
        self.register_buffer("_check_inc", edge_index[0].contiguous(), persistent=False)

        self.check_emb = nn.Linear(d_node_in, h_dim)
        self.var_emb = nn.Linear(d_edge, h_dim)

        edge_feat_dim = h_dim
        self.msg_v2c = nn.Sequential(
            nn.Linear(2 * h_dim, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, edge_feat_dim),
        )
        self.msg_c2v = nn.Sequential(
            nn.Linear(2 * h_dim, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden, edge_feat_dim),
        )

        # GRU: input combines aggregated message + raw initial features
        self.gru_c = nn.GRUCell(edge_feat_dim + d_node_in, h_dim)
        self.gru_v = nn.GRUCell(edge_feat_dim + d_edge, h_dim)

        # No scalars in head
        pool_dim = 2 * h_dim
        self.head = nn.Sequential(
            nn.Linear(pool_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_check = self.n_check
        n_var = self.n_var
        d_in = self.d_node_in
        d_e = self.d_var_in
        h = self.h_dim
        node_block = n_check * d_in
        edge_block = n_var * d_e

        check_init = x[:, :node_block].view(B, n_check, d_in)
        var_init = x[:, node_block : node_block + edge_block].view(B, n_var, d_e)

        h_c = self.check_emb(check_init)
        h_v = self.var_emb(var_init)

        var_inc = self._var_inc
        check_inc = self._check_inc
        E = var_inc.shape[0]

        for _ in range(self.n_iters):
            # Phase A: variable → check
            h_src_v = h_v.index_select(1, var_inc)
            h_dst_c = h_c.index_select(1, check_inc)
            m_v2c = self.msg_v2c(torch.cat([h_src_v, h_dst_c], dim=-1))
            agg_c = torch.zeros(B, n_check, h, device=m_v2c.device, dtype=m_v2c.dtype)
            agg_c.scatter_add_(1, check_inc.view(1, E, 1).expand(B, -1, h), m_v2c)
            gru_in_c = torch.cat([agg_c, check_init], dim=-1)
            h_c = self.gru_c(gru_in_c.reshape(-1, h + d_in),
                             h_c.reshape(-1, h)).view(B, n_check, h)

            # Phase B: check → variable
            h_src_c = h_c.index_select(1, check_inc)
            h_dst_v = h_v.index_select(1, var_inc)
            m_c2v = self.msg_c2v(torch.cat([h_src_c, h_dst_v], dim=-1))
            agg_v = torch.zeros(B, n_var, h, device=m_c2v.device, dtype=m_c2v.dtype)
            agg_v.scatter_add_(1, var_inc.view(1, E, 1).expand(B, -1, h), m_c2v)
            gru_in_v = torch.cat([agg_v, var_init], dim=-1)
            h_v = self.gru_v(gru_in_v.reshape(-1, h + d_e),
                             h_v.reshape(-1, h)).view(B, n_var, h)

        g_check = h_c.mean(dim=1)
        g_var = h_v.mean(dim=1)
        return self.head(torch.cat([g_check, g_var], dim=-1))


@register_model("cached_gap_routed_gnn_cnnfeat")
class CachedGapRoutedGNNMonoCNNFeat(BaseQECModel):
    """Mono GNN over decoding-graph topology + CNN-style aggregated node features.

    Tests information-bottleneck hypothesis: lossless per-edge MWPM info
    (1[E_0], 1[E_1]) is replaced by per-detector aggregated channels
    (matching the CNN paper's 4-channel grid input), forcing the model to
    learn spatial patterns rather than reproduce MWPM trivially.

    Input layout (flat tensor from CachedGapRoutedGNNCNNFeatPreprocessor):
      [node_x (B, n_node * 4)]   — 4 CNN-style channels per detector
      [edge_feat (B, n_undir_e)] — static w_e_norm only (1-dim)
      (scalars REMOVED — no mwpm_pred shortcut)
    """
    REQUIRED_PREPROCESSOR = "cached_gap_routed_gnn_cnnfeat"

    def __init__(
        self,
        num_observables: int,
        n_node: int,
        n_undir_e: int,
        d_node_in: int = 4,
        d_edge: int = 1,
        n_scalars: int = 0,
        edge_index_np=None,
        edge_w_norm_np=None,
        undir_idx_for_dir_np=None,
        d_hidden: int = 64,
        n_layers: int = 3,
        msg_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.2,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        if edge_index_np is None or undir_idx_for_dir_np is None:
            raise ValueError(
                "CachedGapRoutedGNNMonoCNNFeat requires edge_index_np / "
                "undir_idx_for_dir_np from the preprocessor."
            )

        self.n_node = n_node
        self.n_undir_e = n_undir_e
        self.d_node_in = d_node_in
        self.d_edge = d_edge
        self.n_scalars = n_scalars
        self.d_hidden = d_hidden
        self.pool = pool

        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
        undir_idx_for_dir = torch.as_tensor(undir_idx_for_dir_np, dtype=torch.long)
        self.register_buffer("_src", edge_index[0].contiguous(), persistent=False)
        self.register_buffer("_dst", edge_index[1].contiguous(), persistent=False)
        self.register_buffer("_undir_idx_for_dir", undir_idx_for_dir.contiguous(),
                             persistent=False)

        self.node_emb = nn.Linear(d_node_in, d_hidden)
        self.layers = nn.ModuleList([
            EdgeConcatMPNNLayer(d_hidden, d_edge, msg_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        n_node = self.n_node
        n_undir = self.n_undir_e
        d_in = self.d_node_in
        d_e = self.d_edge

        node_block = n_node * d_in
        edge_block = n_undir * d_e

        node_x = x[:, :node_block].view(B, n_node, d_in)                         # (B, n_node, 4)
        edge_undir = x[:, node_block : node_block + edge_block].view(B, n_undir, d_e)

        edge_dir = edge_undir.index_select(1, self._undir_idx_for_dir)            # (B, 2*n_undir, d_e)

        h = self.node_emb(node_x)                                                  # (B, n_node, d_h)
        for layer in self.layers:
            h = layer(h, edge_dir, self._src, self._dst)

        if self.pool == "mean":
            g = h.mean(dim=1)
        elif self.pool == "sum":
            g = h.sum(dim=1)
        elif self.pool == "max":
            g = h.max(dim=1).values
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        return self.head(g)


@register_model("syndrome_gnn_astra_style")
class SyndromeGNNAstraStyle(GapRoutedGNNAstraStyle):
    """ASTRA-style architecture, syndrome-only input (no E_0/E_1 byproducts).

    Pairs with SyndromeGNNPreprocessor (d_edge=1: only w_e_norm as var feature).
    Used as ablation: 'what if we drop MWPM byproducts entirely?'
    """
    REQUIRED_PREPROCESSOR = "syndrome_gnn"

    def __init__(self, num_observables, n_node, n_undir_e,
                 d_edge=1, n_scalars=0, **kwargs):
        super().__init__(
            num_observables=num_observables,
            n_node=n_node, n_undir_e=n_undir_e,
            d_edge=d_edge, n_scalars=n_scalars,
            **kwargs,
        )
