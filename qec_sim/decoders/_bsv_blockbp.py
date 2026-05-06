"""BlockBP partition for BSV PEPS — paper Algorithm 1의 'Partition T_s(L̄) into k×k blocks'.

(2d-1)² 격자 사이트들을 size-k 블록 그리드로 분할. 분할 규칙:
    - 사이트 (x, y) → 블록 (x // k, y // k)
    - 격자 끝에서 k가 cleanly divide 안 하면 가장자리 블록은 더 작음 (paper note 1)
    - 사이트 사이의 bond 중 양 끝 블록이 다르면 cross bond — BP 메시지가 흐를 위치
    - L_acc 텐서는 logical_block(default (0,0))에 할당; bottom-row 데이터 qubit과의
      log_<i> bond 중 자기 데이터 qubit 블록 ≠ logical_block인 것이 cross bond

L_acc는 paper의 lattice-internal logical encoding과 다름 (외부 ghost) — 단순화 용도.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import quimb.tensor as qtn

from ._bsv_peps import (
    _copy_tensor,
    _parity_tensor,
    bond_name,
    bsv_layout,
    neighbors,
)

Coord = Tuple[int, int]
BlockIdx = Tuple[int, int]  # (bi, bj)
DirEdge = Tuple[BlockIdx, BlockIdx]  # (sender, receiver)


@dataclass(frozen=True)
class CrossBond:
    """블록 경계를 가로지르는 bond — BP 메시지가 다닐 위치.

    site_a/site_b는 lattice cross bond에만 의미 있음 (logical cross bond는
    L_acc가 격자 site가 아니라 (0,0) 같은 placeholder).
    """
    bond_name: str
    block_a: BlockIdx
    block_b: BlockIdx
    site_a: Coord = (-1, -1)        # block_a 안에 있는 site (lattice cross bond일 때만)
    site_b: Coord = (-1, -1)        # block_b 안에 있는 site
    is_logical: bool = False         # log_<i> bond이면 True


def side_of_a(cb: CrossBond) -> str:
    """block_a 기준 cross bond가 위치한 변 (xmin/xmax/ymin/ymax). 격자 bond에만 의미.

    site_a → site_b 방향으로 결정:
        site_b = site_a + (-1, 0) → block_a의 xmin (= 왼쪽)
        site_b = site_a + (+1, 0) → xmax
        site_b = site_a + (0, -1) → ymin
        site_b = site_a + (0, +1) → ymax
    """
    if cb.is_logical:
        raise ValueError("logical cross bond는 spatial side가 정의되지 않음")
    dx = cb.site_b[0] - cb.site_a[0]
    dy = cb.site_b[1] - cb.site_a[1]
    if (dx, dy) == (-1, 0):
        return 'xmin'
    if (dx, dy) == (1, 0):
        return 'xmax'
    if (dx, dy) == (0, -1):
        return 'ymin'
    if (dx, dy) == (0, 1):
        return 'ymax'
    raise ValueError(f"non-adjacent cross bond: {cb.site_a} → {cb.site_b}")


@dataclass
class BSVBlockPartition:
    d: int
    k: int
    n: int                                          # = 2d-1
    site_to_block: Dict[Coord, BlockIdx] = field(default_factory=dict)
    block_sites: Dict[BlockIdx, List[Coord]] = field(default_factory=dict)
    lattice_cross_bonds: List[CrossBond] = field(default_factory=list)
    logical_block: BlockIdx = (0, 0)               # L_acc 텐서가 속할 블록
    logical_cross_bonds: List[CrossBond] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.block_sites)

    @property
    def all_blocks(self) -> List[BlockIdx]:
        return sorted(self.block_sites.keys())


def partition_bsv_lattice(d: int, k: int, logical_block: BlockIdx = (0, 0)) -> BSVBlockPartition:
    """(2d-1)² 격자를 k×k 블록으로 분할.

    Args:
        d: 코드 distance
        k: 블록 크기 (≥ 1). 격자 끝에서 cleanly divide 안 하면 가장자리 블록은 작음.
        logical_block: L_acc 텐서를 둘 블록 (default (0,0)).

    Returns:
        BSVBlockPartition. site_to_block, cross_bonds, logical_block 등 채워짐.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}")
    layout = bsv_layout(d)
    n = layout.n
    part = BSVBlockPartition(d=d, k=k, n=n, logical_block=logical_block)

    # 1. 사이트 → 블록
    for x in range(n):
        for y in range(n):
            blk = (x // k, y // k)
            part.site_to_block[(x, y)] = blk
            part.block_sites.setdefault(blk, []).append((x, y))

    # 2. 격자 내부 cross bonds — 인접 사이트 pair 중 다른 블록인 것
    seen_bonds = set()
    for c, blk_a in part.site_to_block.items():
        for nb in neighbors(c, n):
            bn = bond_name(c, nb)
            if bn in seen_bonds:
                continue
            blk_b = part.site_to_block[nb]
            if blk_a != blk_b:
                part.lattice_cross_bonds.append(
                    CrossBond(
                        bond_name=bn, block_a=blk_a, block_b=blk_b,
                        site_a=c, site_b=nb, is_logical=False,
                    )
                )
            seen_bonds.add(bn)

    # 3. logical (log_<i>) bonds — bottom-row 데이터 qubit ↔ L_acc
    for i, dq_coord in enumerate(layout.logical_data):
        dq_block = part.site_to_block[dq_coord]
        if dq_block != logical_block:
            part.logical_cross_bonds.append(
                CrossBond(
                    bond_name=f"log_{i}",
                    block_a=dq_block,
                    block_b=logical_block,
                    site_a=dq_coord, site_b=(-1, -1),
                    is_logical=True,
                )
            )
    # logical_block 자체에 데이터 qubit이 있으면 그쪽 log_<i>는 internal — cross 아님

    return part


# ──────────────────────────────────────────────
# C2: 블록별 텐서 + 단일 메시지 패스 (flat tensor 메시지)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryGroup:
    """두 블록 사이의 모든 cross bonds (lattice + logical) — 한 묶음 메시지.

    bond_names는 메시지 텐서의 인덱스 순서. bonds는 각 bond의 상세 메타 (block-side,
    site_a/site_b, kind). spatial 흐름과 logical 흐름이 같은 (block_a, block_b) pair
    안에 섞여있을 수 있어 (e.g., 인접 블록 + logical_block 경우) 한 group 안에 함께 둠.
    """
    block_a: BlockIdx       # canonical: block_a < block_b
    block_b: BlockIdx
    bond_names: Tuple[str, ...]  # 메시지 텐서의 인덱스 순서 (정렬된 bond_name)
    bonds: Tuple[CrossBond, ...] = ()  # bond_names와 동일 순서, 상세 메타

    @property
    def shape(self) -> Tuple[int, ...]:
        return (2,) * len(self.bond_names)

    def neighbor_of(self, b: BlockIdx) -> BlockIdx:
        if b == self.block_a:
            return self.block_b
        if b == self.block_b:
            return self.block_a
        raise ValueError(f"block {b} not part of this boundary")

    @property
    def has_logical(self) -> bool:
        return any(b.is_logical for b in self.bonds)

    @property
    def all_spatial(self) -> bool:
        return self.bonds and not any(b.is_logical for b in self.bonds)


def boundary_groups(part: BSVBlockPartition) -> List[BoundaryGroup]:
    """모든 cross bonds (격자 + logical) 를 (block_a, block_b) 쌍으로 묶음.

    bond_names는 정렬된 순서. bonds는 그 순서에 맞춰 CrossBond 메타 (canonical
    block_a perspective: cross bond의 block_a/site_a/side가 partition의 그것과
    일치하도록 정렬).
    """
    grouped: Dict[Tuple[BlockIdx, BlockIdx], List[CrossBond]] = {}
    for cb in list(part.lattice_cross_bonds) + list(part.logical_cross_bonds):
        a, b = sorted([cb.block_a, cb.block_b])
        # 만약 cb의 (block_a, block_b)가 canonical 순서와 반대면, site_a/site_b도 swap
        if (cb.block_a, cb.block_b) == (a, b):
            cb_canonical = cb
        else:
            cb_canonical = CrossBond(
                bond_name=cb.bond_name, block_a=a, block_b=b,
                site_a=cb.site_b, site_b=cb.site_a,
                is_logical=cb.is_logical,
            )
        grouped.setdefault((a, b), []).append(cb_canonical)

    out: List[BoundaryGroup] = []
    for (a, b), cb_list in sorted(grouped.items()):
        # bond_name 알파벳 순으로 정렬 (consistent ordering)
        cb_list_sorted = sorted(cb_list, key=lambda x: x.bond_name)
        out.append(BoundaryGroup(
            block_a=a, block_b=b,
            bond_names=tuple(c.bond_name for c in cb_list_sorted),
            bonds=tuple(cb_list_sorted),
        ))
    return out


def block_to_boundaries(
    part: BSVBlockPartition,
    groups: List[BoundaryGroup],
) -> Dict[BlockIdx, List[BoundaryGroup]]:
    """블록 → 자기를 건드리는 BoundaryGroup 목록. partition의 모든 블록을 키로 가짐
    (boundary가 없는 블록은 빈 list)."""
    out: Dict[BlockIdx, List[BoundaryGroup]] = {b: [] for b in part.block_sites}
    for g in groups:
        out[g.block_a].append(g)
        out[g.block_b].append(g)
    return out


def site_2d_tags(part: BSVBlockPartition, c: Coord) -> List[str]:
    """블록 내부 local 좌표 기반 TN2D 태그 셋: `I{x},{y}`, `X{x}`, `Y{y}`."""
    bi, bj = part.site_to_block[c]
    lx = c[0] - bi * part.k
    ly = c[1] - bj * part.k
    return [f"I{lx},{ly}", f"X{lx}", f"Y{ly}"]


def build_block_tensors(
    part: BSVBlockPartition,
    p: float,
    syndrome_at_zstab: Dict[Coord, int],
    class_bit: int,
) -> Dict[BlockIdx, List[qtn.Tensor]]:
    """블록별로 그 블록에 속한 텐서 리스트를 반환. cross bond 인덱스는 양쪽 블록에 같은
    이름으로 노출 (= contract하면 자동으로 합쳐짐). L_acc 텐서는 logical_block에 부착.

    paper Algorithm 1의 'Construct T_s(L̄) ... Partition into k×k blocks' 단계.

    각 격자 사이트 텐서는 `I{local_x},{local_y}` 태그도 가짐 — TN2D 활용 가능.
    L_acc 텐서는 site 좌표가 없어서 'L_acc' 태그만 (TN2D 밖 처리 필요).
    """
    if class_bit not in (0, 1):
        raise ValueError(f"class_bit must be 0 or 1, got {class_bit}")
    layout = bsv_layout(part.d)
    log_idx_of: Dict[Coord, int] = {c: i for i, c in enumerate(layout.logical_data)}

    blocks: Dict[BlockIdx, List[qtn.Tensor]] = {b: [] for b in part.block_sites}

    for c in layout.data_qubits:
        nbrs = neighbors(c, part.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        if c in log_idx_of:
            inds.append(f"log_{log_idx_of[c]}")
        b = part.site_to_block[c]
        blocks[b].append(qtn.Tensor(
            _copy_tensor(len(inds), p), inds,
            tags=[f'D_{c[0]}_{c[1]}', *site_2d_tags(part, c)],
        ))

    for c in layout.z_stabs:
        nbrs = neighbors(c, part.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        s = int(syndrome_at_zstab.get(c, 0)) & 1
        b = part.site_to_block[c]
        blocks[b].append(qtn.Tensor(
            _parity_tensor(len(inds), s), inds,
            tags=[f'Z_{c[0]}_{c[1]}', *site_2d_tags(part, c)],
        ))

    for c in layout.x_stabs:
        nbrs = neighbors(c, part.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        b = part.site_to_block[c]
        blocks[b].append(qtn.Tensor(
            np.ones((2,) * len(inds)), inds,
            tags=[f'X_{c[0]}_{c[1]}', *site_2d_tags(part, c)],
        ))

    # L_acc 텐서: logical_block에 부착, target = class_bit. site 태그 없음.
    log_inds = [f"log_{i}" for i in range(part.d)]
    blocks[part.logical_block].append(
        qtn.Tensor(_parity_tensor(part.d, class_bit), log_inds, tags=['L_acc'])
    )

    return blocks




# ──────────────────────────────────────────────
# C3: BP 루프 + Lemma III.1 (backend-agnostic)
# ──────────────────────────────────────────────

@dataclass
class BPResult:
    messages: Dict[DirEdge, object]      # backend별 다른 타입
    iterations: int
    final_delta: float
    converged: bool                       # Δ < delta_0


def run_bp_loop(
    part: BSVBlockPartition,
    block_tensors: Dict[BlockIdx, List[qtn.Tensor]],
    groups: List[BoundaryGroup],
    boundaries_of: Dict[BlockIdx, List[BoundaryGroup]],
    max_iter: int,
    delta_0: float,
    backend,                              # MessageBackend 구현
    damping: float = 0.0,
) -> BPResult:
    """Synchronous BP loop. paper Algorithm 1의 main loop.

    backend가 메시지 표현/연산을 담당 (FlatBackend / MPSBackend / ...).
    """
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"damping must be in [0, 1), got {damping}")

    msgs: Dict[DirEdge, object] = {}
    for g in groups:
        msgs[(g.block_a, g.block_b)] = backend.normalize(backend.init_uniform(g))
        msgs[(g.block_b, g.block_a)] = backend.normalize(backend.init_uniform(g))

    final_delta = float('inf')
    converged = False
    iters_done = 0

    for t in range(max_iter):
        iters_done = t + 1
        new_msgs: Dict[DirEdge, object] = {}
        for sender in part.block_sites:
            for g in boundaries_of[sender]:
                receiver = g.neighbor_of(sender)
                # target 외 incoming 메시지 모음
                incoming = []
                for og in boundaries_of[sender]:
                    if og is g:
                        continue
                    other = og.neighbor_of(sender)
                    incoming.append((og, msgs[(other, sender)]))
                out = backend.compute_outgoing(
                    sender_block=sender,
                    sender_tensors=block_tensors[sender], target=g, incoming=incoming,
                )
                out = backend.normalize(out)
                if damping > 0:
                    out = backend.normalize(backend.damp(out, msgs[(sender, receiver)], damping))
                new_msgs[(sender, receiver)] = out

        if new_msgs:
            deltas = [backend.delta(new_msgs[k], msgs[k]) for k in new_msgs]
            final_delta = float(np.sqrt(sum(d ** 2 for d in deltas) / len(deltas)))
        else:
            final_delta = 0.0
            converged = True

        msgs = new_msgs
        if final_delta < delta_0:
            converged = True
            break

    return BPResult(
        messages=msgs, iterations=iters_done,
        final_delta=final_delta, converged=converged,
    )


@dataclass
class ClassResult:
    """한 logical class에 대한 BP 결과."""
    class_bit: int
    bp: BPResult
    tr_estimate: float | None  # Δ < delta_1이면 산출, 아니면 None


@dataclass
class DecodeResult:
    predicted_class: int          # argmax 결과
    per_class: List[ClassResult]
    fallback_used: bool           # 둘 다 Δ ≥ delta_1이면 True (Δ_min 폴백)


def decode_blockbp(
    d: int,
    p: float,
    syndrome_at_zstab: Dict[Coord, int],
    k: int,
    max_iter: int,
    delta_0: float,
    delta_1: float,
    backend,
    damping: float = 0.0,
    logical_block: BlockIdx = (0, 0),
) -> DecodeResult:
    """Paper Algorithm 1의 binary 버전 (bit-flip Z-memory).

    각 class_bit ∈ {0, 1}에 대해 BSV TN 빌드 → partition → BP loop 돌림.
    수렴된(Δ < delta_1) class들 중 Tr 최대인 것 선택. 둘 다 미수렴이면 Δ 최소인 것.

    `backend`: MessageBackend 구현 (FlatBackend / MPSBackend / ...). χ 같은 표현
        파라미터는 backend 생성 시 결정.

    delta_0 < delta_1: delta_0은 BP 루프 early-exit 임계값, delta_1은 결과 신뢰 임계값.
    """
    if not (delta_0 <= delta_1):
        raise ValueError(f"delta_0 ({delta_0}) must be <= delta_1 ({delta_1})")

    part = partition_bsv_lattice(d=d, k=k, logical_block=logical_block)
    groups = boundary_groups(part)
    boundaries_of = block_to_boundaries(part, groups)

    per_class: List[ClassResult] = []
    for class_bit in (0, 1):
        block_tensors = build_block_tensors(
            part, p=p, syndrome_at_zstab=syndrome_at_zstab, class_bit=class_bit,
        )
        bp = run_bp_loop(
            part=part, block_tensors=block_tensors, groups=groups,
            boundaries_of=boundaries_of, max_iter=max_iter, delta_0=delta_0,
            backend=backend, damping=damping,
        )
        tr = (
            estimate_tr_from_messages(part, block_tensors, groups, boundaries_of,
                                      bp.messages, backend)
            if bp.final_delta < delta_1
            else None
        )
        per_class.append(ClassResult(class_bit=class_bit, bp=bp, tr_estimate=tr))

    converged = [c for c in per_class if c.tr_estimate is not None]
    if converged:
        winner = max(converged, key=lambda c: c.tr_estimate)
        fallback = False
    else:
        winner = min(per_class, key=lambda c: c.bp.final_delta)
        fallback = True

    return DecodeResult(
        predicted_class=winner.class_bit,
        per_class=per_class,
        fallback_used=fallback,
    )


def estimate_tr_from_messages(
    part: BSVBlockPartition,
    block_tensors: Dict[BlockIdx, List[qtn.Tensor]],
    groups: List[BoundaryGroup],
    boundaries_of: Dict[BlockIdx, List[BoundaryGroup]],
    messages: Dict[DirEdge, object],
    backend,
) -> float:
    """paper Lemma III.1: Tr(T) ≃ ∏_v Tr(T_v ∏_{u∈N_v} m̂_{u→v}).

    backend가 inner / scale / inject 연산 제공.

    Steps:
      1. 각 boundary 쌍 (u→v, v→u)을 backend.inner(m̂_{u→v}, m̂_{v→u}) = 1 이도록 rescale
      2. 블록 v별로: 자기 텐서 + 모든 incoming m̂_{u→v} (backend.inject) contract → scalar
      3. 모든 블록의 scalar 곱 = Tr 추정값
    """
    rescaled: Dict[DirEdge, object] = {}
    for g in groups:
        a, b = g.block_a, g.block_b
        m_ab = messages[(a, b)]
        m_ba = messages[(b, a)]
        ip = backend.inner(m_ab, m_ba)
        if ip > 0:
            s = 1.0 / np.sqrt(ip)
        elif ip < 0:
            s = 1.0 / np.sqrt(abs(ip))
        else:
            s = 1.0
        rescaled[(a, b)] = backend.scale(m_ab, s)
        rescaled[(b, a)] = backend.scale(m_ba, s)

    total = 1.0
    for blk in part.block_sites:
        tensors = list(block_tensors[blk])
        for g in boundaries_of[blk]:
            other = g.neighbor_of(blk)
            tensors.extend(backend.inject(rescaled[(other, blk)], g))
        sub_tn = qtn.TensorNetwork(tensors)
        total *= float(sub_tn.contract())
    return total
