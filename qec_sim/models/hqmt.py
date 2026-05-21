"""HQMT — Hierarchical Qubit-Merging Transformer (Park et al., arXiv:2510.11593v1).

2-stage transformer decoder with qubit-merging layer and Stage 1/2 weight sharing.
Designed for code-capacity (single-round, perfect syndrome) depolarizing noise.

Stage 1: 2n Z/X tokens → N transformer blocks (fine-grained)
Merge  : per-qubit concat [Z; X] → Linear(2D → D) → n qubit tokens
Stage 2: n tokens → same N transformer blocks (weight-shared with Stage 1)
Output : mean-pool → Linear(D → num_obs)

Reuses `qct_qubit_centric` preprocessor for topology (nbr_Z, nbr_X, det_Z_at,
det_X_at). For multi-round circuits, the LAST round of each stab is used.
For paper-faithful HQMT use a code-capacity circuit with T_Z == T_X == 1.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = x.shape
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class _QubitMergingLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x_z: torch.Tensor, x_x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([x_z, x_x], dim=-1))


@register_model("hqmt")
class HQMT(BaseQECModel):
    """HQMT decoder (Park et al., arXiv:2510.11593v1, 2-stage hierarchical)."""

    REQUIRED_PREPROCESSOR = "qct_qubit_centric"

    def __init__(
        self,
        num_observables: int,
        # Topology from qct_qubit_centric preprocessor
        n: int, m_Z: int, m_X: int, T_Z: int, T_X: int, K_Z: int, K_X: int,
        nbr_Z: torch.Tensor, nbr_X: torch.Tensor,
        det_Z_at: torch.Tensor, det_X_at: torch.Tensor,
        attn_mask: torch.Tensor,
        # Hyperparameters (paper v1 default)
        d_model: int = 128,
        n_heads: int = 8,
        N: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        self.n = n
        self.m_Z = m_Z
        self.m_X = m_X
        self.d_model = d_model
        self.N = N

        if T_Z < 1 or T_X < 1:
            raise ValueError(
                f"HQMT needs both Z and X detectors. Got T_Z={T_Z}, T_X={T_X}. "
                "Provide a code-capacity circuit with single-round Z+X syndrome."
            )

        # For code-capacity (T_τ == 1), this is the only round.
        # For multi-round circuits we take the LAST round (final perfect measurement),
        # which is the closest analog but not paper-faithful.
        self.register_buffer("z_det_idx", det_Z_at[:, -1].long(), persistent=False)
        self.register_buffer("x_det_idx", det_X_at[:, -1].long(), persistent=False)

        # Build (n × m_τ) adjacency mask from neighbor index lists.
        z_mask = torch.zeros(n, m_Z, dtype=torch.float32)
        x_mask = torch.zeros(n, m_X, dtype=torch.float32)
        for qi in range(n):
            for k in range(K_Z):
                ai = int(nbr_Z[qi, k].item())
                if ai >= 0:
                    z_mask[qi, ai] = 1.0
            for k in range(K_X):
                ai = int(nbr_X[qi, k].item())
                if ai >= 0:
                    x_mask[qi, ai] = 1.0
        self.register_buffer("z_adjacency_mask", z_mask, persistent=False)
        self.register_buffer("x_adjacency_mask", x_mask, persistent=False)

        d_ff = ffn_mult * d_model

        self.z_embedding = nn.Linear(m_Z, d_model)
        self.x_embedding = nn.Linear(m_X, d_model)

        # Stage 1 과 Stage 2 가 동일한 N 개 블록을 공유 (paper v1)
        self.shared_transformer = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(N)
        ])

        self.qubit_merging = _QubitMergingLayer(d_model)

        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_observables)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_patches(self, syndrome: torch.Tensor):
        s_z = syndrome[:, self.z_det_idx]
        s_x = syndrome[:, self.x_det_idx]
        v_z = 1.0 - 2.0 * s_z
        v_x = 1.0 - 2.0 * s_x
        p_z = v_z.unsqueeze(1) * self.z_adjacency_mask.unsqueeze(0)
        p_x = v_x.unsqueeze(1) * self.x_adjacency_mask.unsqueeze(0)
        return p_z, p_x

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        p_z, p_x = self._build_patches(syndrome)

        z_tokens = self.z_embedding(p_z)
        x_tokens = self.x_embedding(p_x)

        X1 = torch.cat([z_tokens, x_tokens], dim=1)  # (B, 2n, D)
        h = X1
        for block in self.shared_transformer:
            h = block(h)

        h_z = h[:, :self.n, :]
        h_x = h[:, self.n:, :]
        X2 = self.qubit_merging(h_z, h_x)            # (B, n, D)

        h = X2
        for block in self.shared_transformer:
            h = block(h)

        h = self.output_norm(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


@register_model("hqmt_multiround")
class HQMTMultiRound(BaseQECModel):
    """HQMT (Park et al. v1) extended to multi-round SI1000.

    Combines HQMT's 2-stage + weight-shared transformer with QCT-style
    per-round input embedding: each round's per-qubit patch is embedded
    independently (shared per-stab-type weights), per-round positional
    embed is added, rounds are concatenated, and a merge Linear collapses
    to the d_model token. The rest of the Stage 1 / Merging / Stage 2
    pipeline is identical to HQMT v1.

    For T_Z == T_X == 1, this is equivalent to HQMT v1 up to the extra
    merge Linear (which becomes Identity-like since the concat dim equals
    d_model). Useful for fair SI1000 comparison.
    """

    REQUIRED_PREPROCESSOR = "qct_qubit_centric"

    def __init__(
        self,
        num_observables: int,
        n: int, m_Z: int, m_X: int, T_Z: int, T_X: int, K_Z: int, K_X: int,
        nbr_Z: torch.Tensor, nbr_X: torch.Tensor,
        det_Z_at: torch.Tensor, det_X_at: torch.Tensor,
        attn_mask: torch.Tensor,
        d_model: int = 128,
        n_heads: int = 8,
        N: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        self.n = n
        self.m_Z = m_Z
        self.m_X = m_X
        self.T_Z = T_Z
        self.T_X = T_X
        self.d_model = d_model
        self.N = N

        if T_Z < 1 or T_X < 1:
            raise ValueError(
                f"HQMTMultiRound needs T_Z>=1, T_X>=1, got T_Z={T_Z}, T_X={T_X}.")

        # All rounds of detectors (not just the last)
        self.register_buffer("det_Z_at", det_Z_at.long(), persistent=False)
        self.register_buffer("det_X_at", det_X_at.long(), persistent=False)

        # Adjacency masks (n × m_τ)
        z_mask = torch.zeros(n, m_Z, dtype=torch.float32)
        x_mask = torch.zeros(n, m_X, dtype=torch.float32)
        for qi in range(n):
            for k in range(K_Z):
                ai = int(nbr_Z[qi, k].item())
                if ai >= 0:
                    z_mask[qi, ai] = 1.0
            for k in range(K_X):
                ai = int(nbr_X[qi, k].item())
                if ai >= 0:
                    x_mask[qi, ai] = 1.0
        self.register_buffer("z_adjacency_mask", z_mask, persistent=False)
        self.register_buffer("x_adjacency_mask", x_mask, persistent=False)

        d_ff = ffn_mult * d_model

        # Per-round embedding (shared across rounds, per stab type)
        self.z_embedding = nn.Linear(m_Z, d_model)
        self.x_embedding = nn.Linear(m_X, d_model)

        # Per-round learnable positional embed (QCT-style)
        self.pos_round_Z = nn.Parameter(torch.zeros(T_Z, d_model))
        self.pos_round_X = nn.Parameter(torch.zeros(T_X, d_model))
        nn.init.trunc_normal_(self.pos_round_Z, std=0.02)
        nn.init.trunc_normal_(self.pos_round_X, std=0.02)

        # Round-concat merge: (T_τ × d_model) → d_model
        self.merge_rounds_Z = nn.Linear(T_Z * d_model, d_model)
        self.merge_rounds_X = nn.Linear(T_X * d_model, d_model)

        # Stage 1/2 shared transformer (v1 paper)
        self.shared_transformer = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(N)
        ])

        self.qubit_merging = _QubitMergingLayer(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_observables)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_per_round_tokens(self, syndrome: torch.Tensor):
        """syndrome: (B, num_detectors) → (z_tokens, x_tokens) each (B, n, D)."""
        # Per-stab, per-round signs (B, m_τ, T_τ)
        v = 1.0 - 2.0 * syndrome
        v_z = v[:, self.det_Z_at]   # (B, m_Z, T_Z)
        v_x = v[:, self.det_X_at]   # (B, m_X, T_X)

        # Per-qubit, per-round patches: (B, n, T_τ, m_τ) via adjacency broadcast.
        # patch[b,i,t,a] = v[b,a,t] * mask[i,a]
        p_z = v_z.permute(0, 2, 1).unsqueeze(1) * self.z_adjacency_mask.unsqueeze(0).unsqueeze(2)
        p_x = v_x.permute(0, 2, 1).unsqueeze(1) * self.x_adjacency_mask.unsqueeze(0).unsqueeze(2)
        # shapes: (B, n, T_τ, m_τ)

        # Embed per round (shared weights), add positional
        e_z = self.z_embedding(p_z) + self.pos_round_Z  # (B, n, T_Z, D)
        e_x = self.x_embedding(p_x) + self.pos_round_X  # (B, n, T_X, D)

        # Concat over rounds → (B, n, T_τ × D), merge → (B, n, D)
        z_tokens = self.merge_rounds_Z(e_z.flatten(start_dim=2))
        x_tokens = self.merge_rounds_X(e_x.flatten(start_dim=2))
        return z_tokens, x_tokens

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        z_tokens, x_tokens = self._build_per_round_tokens(syndrome)

        X1 = torch.cat([z_tokens, x_tokens], dim=1)  # (B, 2n, D)
        h = X1
        for block in self.shared_transformer:
            h = block(h)

        h_z = h[:, :self.n, :]
        h_x = h[:, self.n:, :]
        X2 = self.qubit_merging(h_z, h_x)            # (B, n, D)

        h = X2
        for block in self.shared_transformer:
            h = block(h)

        h = self.output_norm(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
