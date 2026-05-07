"""Qubit-centric Transformer (QCT) for surface code decoding.

Park, Kwak, Kim — "Qubit-centric Transformer for Surface Code Decoding"
arXiv:2510.11593 (Mar 2026).

Paper-faithful 구현. 단, paper는 code-capacity depolarizing (single round, perfect syndrome,
4-class logical class) — 본 구현은 SI1000 multi-round detector 입력에 맞춰 다음을 추가:
  - 시간 차원: stab type τ별로 detector가 존재하는 round를 따로 모아 T_Z, T_X 정의
  - Embedding은 round t 별로 ξ^τ_{i,t}를 만들고 W_τ로 D_e 차원에 사상한 뒤 round 차원으로
    concat → φ^τ_i ∈ R^{T_τ·D_e}. positional embed는 (qubit, stab type, round) 모두 학습 가능.
  - 출력 dim은 SI1000 Z-memory에서는 num_observables=1 (binary) 또는 2 (coset_mode 2-class).
"""

import torch
import torch.nn as nn

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


@register_model("qct")
class QCT(BaseQECModel):
    REQUIRED_PREPROCESSOR = "qct_qubit_centric"

    def __init__(
        self,
        num_observables: int,
        # Topology (preprocessor가 주입)
        n: int, m_Z: int, m_X: int, T_Z: int, T_X: int, K_Z: int, K_X: int,
        nbr_Z: torch.Tensor, nbr_X: torch.Tensor,
        det_Z_at: torch.Tensor, det_X_at: torch.Tensor,
        attn_mask: torch.Tensor,
        # 하이퍼 (paper default: D_e=64, D_m=128, N=8)
        D_e: int = 64,
        D_m: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
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
        self.K_Z = K_Z
        self.K_X = K_X
        self.D_e = D_e
        self.D_m = D_m

        # Topology buffers (학습 안 됨)
        self.register_buffer("nbr_Z", nbr_Z.long(), persistent=False)
        self.register_buffer("nbr_X", nbr_X.long(), persistent=False)
        self.register_buffer("det_Z_at", det_Z_at.long(), persistent=False)
        self.register_buffer("det_X_at", det_X_at.long(), persistent=False)
        # attn_mask: True = unmasked, False = masked. float mask로 변환해서 저장.
        attn_float = torch.zeros_like(attn_mask, dtype=torch.float32)
        attn_float.masked_fill_(~attn_mask, float("-inf"))
        self.register_buffer("attn_float_mask", attn_float, persistent=False)

        # Padding sentinel: nbr_*에서 -1을 0으로 바꾼 안전 인덱스 + valid mask
        nbr_Z_safe = nbr_Z.clamp(min=0)
        nbr_X_safe = nbr_X.clamp(min=0)
        valid_Z = (nbr_Z >= 0).float().unsqueeze(-1)  # (n, K_Z, 1)
        valid_X = (nbr_X >= 0).float().unsqueeze(-1)
        self.register_buffer("nbr_Z_safe", nbr_Z_safe.long(), persistent=False)
        self.register_buffer("nbr_X_safe", nbr_X_safe.long(), persistent=False)
        self.register_buffer("valid_Z", valid_Z, persistent=False)
        self.register_buffer("valid_X", valid_X, persistent=False)

        # Embedding: W_τ ∈ R^{D_e × m_τ}
        self.W_Z = nn.Linear(m_Z, D_e)
        self.W_X = nn.Linear(m_X, D_e)
        # 학습 가능한 positional embed: (qubit, stab type) + (round, stab type)
        # per-qubit: paper의 p^τ_i (단, paper는 round 없음 — 본 구현은 둘 다 도입)
        self.pos_qubit_Z = nn.Parameter(torch.zeros(n, D_e))
        self.pos_qubit_X = nn.Parameter(torch.zeros(n, D_e))
        self.pos_round_Z = nn.Parameter(torch.zeros(T_Z, D_e))
        self.pos_round_X = nn.Parameter(torch.zeros(T_X, D_e))
        nn.init.trunc_normal_(self.pos_qubit_Z, std=0.02)
        nn.init.trunc_normal_(self.pos_qubit_X, std=0.02)
        nn.init.trunc_normal_(self.pos_round_Z, std=0.02)
        nn.init.trunc_normal_(self.pos_round_X, std=0.02)

        # Merging: concat [φ^Z; φ^X] → D_m
        self.merge = nn.Linear((T_Z + T_X) * D_e, D_m)

        # Transformer: pre-LN (paper figure: LN → SA → LN → FFN)
        layer = nn.TransformerEncoderLayer(
            d_model=D_m, nhead=num_heads,
            dim_feedforward=ffn_mult * D_m,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Output: mean-pool n tokens → FC → num_observables logits
        self.head = nn.Linear(D_m, num_observables)

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """syndromes: (B, num_detectors) in {0, 1} as float."""
        B = syndromes.shape[0]

        # σ_j = (-1)^{s_j}  →  0 → +1, 1 → -1
        sigma = 1.0 - 2.0 * syndromes  # (B, num_detectors)

        # Per-stab per-round 신호 (B, m_τ, T_τ)
        sigma_Z = sigma[:, self.det_Z_at]   # (B, m_Z, T_Z)
        sigma_X = sigma[:, self.det_X_at]   # (B, m_X, T_X)

        # 큐빗별 인접 stab gather: (B, n, K_τ, T_τ)
        sigma_Z_q = sigma_Z[:, self.nbr_Z_safe, :] * self.valid_Z   # (B, n, K_Z, T_Z)
        sigma_X_q = sigma_X[:, self.nbr_X_safe, :] * self.valid_X

        # W_τ.weight 형상: (D_e, m_τ). 큐빗별 인접 stab의 임베딩 컬럼 모음:
        # W_Z_nbr[i, k, :] = W_Z.weight[:, nbr_Z_safe[i, k]]
        W_Z_nbr = self.W_Z.weight[:, self.nbr_Z_safe].permute(1, 2, 0)  # (n, K_Z, D_e)
        W_X_nbr = self.W_X.weight[:, self.nbr_X_safe].permute(1, 2, 0)

        # φ^τ_{b, i, t, :} = Σ_k sigma_q[b,i,k,t] * W_τ_nbr[i,k,:]
        phi_Z = torch.einsum("bikt,ikd->bitd", sigma_Z_q, W_Z_nbr)  # (B, n, T_Z, D_e)
        phi_X = torch.einsum("bikt,ikd->bitd", sigma_X_q, W_X_nbr)

        # bias + positional (qubit, round)
        # bias (D_e,) — broadcast 마지막 dim
        # pos_qubit_τ (n, D_e) — broadcast (1, n, 1, D_e)
        # pos_round_τ (T_τ, D_e) — broadcast (1, 1, T_τ, D_e)
        phi_Z = (phi_Z
                 + self.W_Z.bias
                 + self.pos_qubit_Z.unsqueeze(0).unsqueeze(2)
                 + self.pos_round_Z.unsqueeze(0).unsqueeze(0))
        phi_X = (phi_X
                 + self.W_X.bias
                 + self.pos_qubit_X.unsqueeze(0).unsqueeze(2)
                 + self.pos_round_X.unsqueeze(0).unsqueeze(0))

        # round 차원 concat
        phi_Z_cat = phi_Z.flatten(start_dim=2)  # (B, n, T_Z * D_e)
        phi_X_cat = phi_X.flatten(start_dim=2)  # (B, n, T_X * D_e)

        # Merging
        u = torch.cat([phi_Z_cat, phi_X_cat], dim=-1)  # (B, n, (T_Z+T_X)*D_e)
        x = self.merge(u)  # (B, n, D_m)

        # Transformer with structure-aware mask
        x = self.transformer(x, mask=self.attn_float_mask)

        # mean-pool over n tokens → head
        z = x.mean(dim=1)  # (B, D_m)
        return self.head(z)
