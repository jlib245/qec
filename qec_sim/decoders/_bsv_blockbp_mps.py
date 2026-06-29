"""BSV BlockBP decoder using BlockBPMps (paper-faithful BP on closed per-class TN).

각 logical class에 대해:
    1. `build_bsv_tn_for_class` 로 closed TN 빌드 (현재 L_acc ghost 포함)
    2. site_tag로 (x // k, y // k) 블록 분할 — L_acc는 logical_block(default (0,0))에
    3. `contract_blockbp_mps` 로 BP 결과값 (= π(f_s L̄ G^X) 추정) 산출

argmax → predicted class. 두 클래스 모두 BP 미수렴 시 fallback_used=True.

기존 `decode_blockbp` (`_bsv_blockbp.py`)와 다른 점:
    - BP 알고리즘이 자체 backend (FlatBackend/MPSBackend) 대신 quimb BPC 위 BlockBPMps
    - dispatch 분기 없음 (L_acc는 그냥 한 블록에 site_tag로 들어감)
    - dense (`max_chi=None`) / MPS (`max_chi=int`) 통일 인터페이스
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import quimb.tensor as qtn

from ._blockbp_mps import contract_blockbp_mps
from ._bsv_peps import build_bsv_tn_for_class

Coord = Tuple[int, int]


@dataclass
class BlockBPMpsDecodeResult:
    """단일 syndrome decoding 결과."""
    predicted_class: int
    z_values: Tuple[float, float]              # (Z_class0, Z_class1)
    iterations: Tuple[int, int]
    converged: Tuple[bool, bool]
    fallback_used: bool                        # 둘 다 미수렴 = True


def add_block_tags_bsv(
    tn: qtn.TensorNetwork,
    k: int,
    logical_block_tag: str = "B_0_0",
) -> Tuple[qtn.TensorNetwork, List[str]]:
    """BSV 텐서들에 site_tag 부여.

    좌표 기반 (D_x_y / Z_x_y / X_x_y) → `B_{x//k}_{y//k}`. L_acc는 logical_block_tag.
    Returns (tagged_tn, sorted_site_tags).
    """
    new_tensors = []
    for t in tn.tensors:
        coord: Optional[Coord] = None
        for tag in t.tags:
            if isinstance(tag, str) and len(tag) > 2 and tag[1] == "_":
                kind = tag[0]
                if kind in ("D", "Z", "X"):
                    parts = tag[2:].split("_")
                    if len(parts) == 2:
                        try:
                            coord = (int(parts[0]), int(parts[1]))
                            break
                        except ValueError:
                            pass
        if coord is not None:
            stag = f"B_{coord[0] // k}_{coord[1] // k}"
        elif "L_acc" in t.tags:
            stag = logical_block_tag
        else:
            raise ValueError(f"untagged tensor: {t.tags}")
        new_t = t.copy()
        new_t.add_tag(stag)
        new_tensors.append(new_t)
    site_tags = sorted({
        s for tt in new_tensors
        for s in tt.tags if isinstance(s, str) and s.startswith("B_")
    })
    return qtn.TensorNetwork(new_tensors), site_tags


def decode_blockbp_mps(
    d: int,
    p: float,
    syndrome_at_zstab: Dict[Coord, int],
    k: int,
    *,
    max_chi: Optional[int] = None,
    max_iter: int = 200,
    tol: float = 1e-7,
    damping: float = 0.3,
    update: str = "sequential",
    optimize: str = "auto-hq",
    expr_cache: Optional[Dict] = None,
    n_workers: Optional[int] = None,
) -> BlockBPMpsDecodeResult:
    """Paper Algorithm 1 (binary, bit-flip Z-memory) — BlockBPMps 기반.

    각 class_bit ∈ {0, 1}에 대해 closed TN → BlockBPMps contract → Z value 산출.
    argmax → predicted_class. 두 클래스 모두 미수렴이면 그대로 argmax 적용
    (fallback_used=True로 표시).

    Args:
        max_chi: None이면 dense BP (vanilla l1bp 동등), int이면 MPS chi truncation.
        max_iter, tol, damping: BP 수렴 파라미터. 기존 decode_blockbp와 동일 의미.
        update, optimize: BlockBPMps 옵션.
        expr_cache: cotengra 표현식 cache (cross-shot 재사용용). Session 객체로 관리.
    """
    z_values: List[float] = []
    iterations: List[int] = []
    converged: List[bool] = []

    for class_bit in (0, 1):
        tn = build_bsv_tn_for_class(
            d=d, p=p, syndrome_at_zstab=syndrome_at_zstab, class_bit=class_bit,
        )
        tn_tagged, site_tags = add_block_tags_bsv(tn, k=k)
        info: Dict = {}
        z = contract_blockbp_mps(
            tn_tagged, site_tags=site_tags,
            max_iterations=max_iter, tol=tol, damping=damping,
            max_chi=max_chi, update=update, optimize=optimize,
            info=info, expr_cache=expr_cache, n_workers=n_workers,
        )
        z_values.append(float(z))
        iterations.append(int(info.get("iterations", max_iter)))
        converged.append(bool(info.get("converged", False)))

    predicted = int(np.argmax(z_values))
    fallback_used = not all(converged)

    return BlockBPMpsDecodeResult(
        predicted_class=predicted,
        z_values=tuple(z_values),
        iterations=tuple(iterations),
        converged=tuple(converged),
        fallback_used=fallback_used,
    )


@dataclass
class BlockBPMpsSession:
    """같은 (d, k, partition, max_chi)에서 여러 syndrome 디코딩 시 cotengra 표현식 재사용.

    BSV TN 구조는 syndrome 데이터와 무관 (Z-stab parity 텐서의 값만 변함). 따라서
    `_compute_outgoing`의 cache key (블록 ID + 텐서 shape + canonical bond names) 는
    syndrome 무관하게 stable. 한 session 내에서 expr_cache 공유 → 매 shot마다 cotengra
    path 재계산 안 함.

    사용:
        session = BlockBPMpsSession()
        for syndrome in syndromes:
            res = session.decode(d=d, p=p, syndrome_at_zstab=syndrome, k=3, max_chi=8)
    """
    expr_cache: Dict = field(default_factory=dict)

    def decode(
        self,
        d: int,
        p: float,
        syndrome_at_zstab: Dict[Coord, int],
        k: int,
        *,
        max_chi: Optional[int] = None,
        max_iter: int = 200,
        tol: float = 1e-7,
        damping: float = 0.3,
        update: str = "sequential",
        optimize: str = "auto-hq",
        n_workers: Optional[int] = None,
    ) -> BlockBPMpsDecodeResult:
        return decode_blockbp_mps(
            d=d, p=p, syndrome_at_zstab=syndrome_at_zstab, k=k,
            max_chi=max_chi, max_iter=max_iter, tol=tol, damping=damping,
            update=update, optimize=optimize,
            expr_cache=self.expr_cache, n_workers=n_workers,
        )

    def cache_size(self) -> int:
        return len(self.expr_cache)

    def clear_cache(self):
        self.expr_cache.clear()
