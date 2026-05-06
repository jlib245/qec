"""BlockBP의 message backend 추상.

paper Eq. (6) BP rule은 메시지 표현(flat 텐서, MPS, density matrix, ...)와 무관함.
이 모듈은 그 추상을 정의 — BP 루프는 `MessageBackend`만 호출하고 구체 표현엔 무지.

현재 구현:
    FlatBackend — 메시지가 (2,)^N flat 텐서. 작은 d / 짧은 boundary에 효율.

후속 구현:
    MPSBackend — 메시지가 quimb MPS, χ 로 SVD truncation. paper 본 알고리즘.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Tuple, runtime_checkable

import cotengra as ctg
import numpy as np
import quimb.tensor as qtn

from ._bsv_blockbp import BoundaryGroup


# 메시지 타입은 backend별로 다름 — Protocol에서 generic으로 표현 안 함 (단순화)
Message = object


@runtime_checkable
class MessageBackend(Protocol):
    """BP 메시지의 표현 + 연산을 캡슐화. flat / MPS / 등 다양한 표현 지원."""

    def init_uniform(self, boundary: BoundaryGroup) -> Message:
        """uniform (정규화 안 됨) 초기 메시지 생성."""
        ...

    def compute_outgoing(
        self,
        sender_block,                     # BlockIdx — sender block 좌표
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, Message]],
    ) -> Message:
        """sender 블록 + (target 외) incoming 메시지들 → target boundary로 나가는 메시지.

        paper Eq. (6) BP rule. 내부적으로 SVD truncation 적용 (backend별 χ).
        """
        ...

    def normalize(self, m: Message) -> Message:
        """Unit norm 정규화."""
        ...

    def damp(self, new: Message, old: Message, eta: float) -> Message:
        """paper Eq. (10): m_damp = (1-η) m_new + η m_old."""
        ...

    def delta(self, new: Message, old: Message) -> float:
        """수렴 척도 (paper Eq. 9 기반)."""
        ...

    def inner(self, m_ab: Message, m_ba: Message) -> float:
        """Lemma III.1 rescaling용 — Tr(m_ab · m_ba)."""
        ...

    def scale(self, m: Message, factor: float) -> Message:
        """스칼라 배."""
        ...

    def inject(self, m: Message, boundary: BoundaryGroup) -> List[qtn.Tensor]:
        """블록 contract 시 메시지를 텐서들로 변환 (block + 메시지 → sub-TN)."""
        ...


# ──────────────────────────────────────────────
# Flat tensor backend
# ──────────────────────────────────────────────

class FlatBackend:
    """메시지가 (2,)^N 짜리 flat numpy 텐서. N = boundary cross bond 수.

    작은 d / boundary가 작을 때(N ≤ 6~7) 메모리/시간 모두 효율적.
    `max_bond`은 블록 내부 contract 시 SVD truncation 한계 (메시지 크기 자체는 2^N 고정).
    """

    def __init__(self, max_bond: int):
        if max_bond < 1:
            raise ValueError(f"max_bond must be >= 1, got {max_bond}")
        self.max_bond = max_bond
        # (sender_block, target.bond_names) → cotengra contraction expression.
        # 한 BP loop 동안 sender_tensors 구조와 incoming 순서가 불변이라 path 캐시 안전.
        # block 내부 bonds는 모두 dim 2이고 d=3 k=3 등 작은 블록에선 chi truncation이
        # 효과 없어 plain contract로 처리 (max_bond는 남겨두지만 사용 안 함).
        self._expr_cache: Dict[Tuple, object] = {}

    def init_uniform(self, boundary: BoundaryGroup) -> np.ndarray:
        return np.ones(boundary.shape, dtype=np.float64)

    def compute_outgoing(
        self,
        sender_block,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, np.ndarray]],
    ) -> np.ndarray:
        cache_key = (sender_block, target.bond_names,
                     tuple(g.bond_names for g, _ in incoming))
        expr = self._expr_cache.get(cache_key)
        if expr is None:
            inputs = [tuple(t.inds) for t in sender_tensors]
            shapes = [tuple(t.shape) for t in sender_tensors]
            for g, _ in incoming:
                inputs.append(tuple(g.bond_names))
                shapes.append(g.shape)
            expr = ctg.array_contract_expression(
                inputs=inputs,
                output=tuple(target.bond_names),
                shapes=shapes,
                optimize='greedy',
            )
            self._expr_cache[cache_key] = expr

        arrays = [t.data for t in sender_tensors]
        for _, msg in incoming:
            arrays.append(msg)
        return np.ascontiguousarray(expr(*arrays))

    def normalize(self, m: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(m))
        return m / n if n > 0 else m

    def damp(self, new: np.ndarray, old: np.ndarray, eta: float) -> np.ndarray:
        return (1 - eta) * new + eta * old

    def delta(self, new: np.ndarray, old: np.ndarray) -> float:
        return float(np.linalg.norm(new - old))

    def inner(self, m_ab: np.ndarray, m_ba: np.ndarray) -> float:
        return float(np.tensordot(m_ab, m_ba, axes=m_ab.ndim))

    def scale(self, m: np.ndarray, factor: float) -> np.ndarray:
        return m * factor

    def inject(self, m: np.ndarray, boundary: BoundaryGroup) -> List[qtn.Tensor]:
        return [qtn.Tensor(m, list(boundary.bond_names), tags=['msg_in'])]


# ──────────────────────────────────────────────
# MPS backend (paper-faithful)
# ──────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class MPSMessage:
    """boundary bond_names를 physical indices로 갖는 MPS 메시지.

    mps가 None인 경우는 boundary가 비어있는 degenerate (단일 블록 partition).
    """
    mps: object  # qtn.MatrixProductState (None for degenerate)
    bond_names: tuple


class MPSBackend:
    """메시지가 quimb MPS — boundary chain 위 MPS, virtual bond dim ≤ chi.

    paper Sec III B의 메시지 표현 (boundary MPS). 큰 d에서 메모리 효율 (boundary가 길어도
    chi로 cap). 작은 d에선 FlatBackend보다 약간 느릴 수 있음 (SVD 오버헤드).

    구현:
        init_uniform — uniform tensor → from_dense
        compute_outgoing — sender block + injected MPS sites contract → flat result →
            from_dense(max_bond=chi)로 MPS 형식 복원
        normalize / damp / inner / scale — quimb MPS 메서드
        inject — MPS site 텐서들을 그대로 추가
    """

    def __init__(self, chi: int, max_bond_block: int = None):
        if chi < 1:
            raise ValueError(f"chi must be >= 1, got {chi}")
        self.chi = chi
        # 블록 내부 contract 시 SVD truncation 한계 (다른 chi일 수도)
        self.max_bond_block = max_bond_block if max_bond_block is not None else max(chi, 8)
        # _compute_outgoing_flat 의 path 캐시. 키는
        # (sender_block, target.bond_names, sender_inds, incoming_canonical_meta).
        # incoming MPS의 virtual bond는 canonical 이름으로 매핑 후 키에 포함.
        # shape 변동 (chi truncation 결과) 시 새 entry — BP warmup 후엔 hit.
        self._expr_cache: Dict[Tuple, object] = {}

    def _build_mps(self, arr: np.ndarray, bond_names: tuple, max_bond: int):
        """dense array → MPS with physical indices = bond_names."""
        N = len(bond_names)
        mps = qtn.MatrixProductState.from_dense(arr, dims=(2,) * N, max_bond=max_bond)
        mps.reindex({f'k{i}': bn for i, bn in enumerate(bond_names)}, inplace=True)
        return mps

    def init_uniform(self, boundary: BoundaryGroup) -> MPSMessage:
        N = len(boundary.bond_names)
        if N == 0:
            return MPSMessage(mps=None, bond_names=())
        arr = np.ones((2,) * N)
        mps = self._build_mps(arr, boundary.bond_names, max_bond=self.chi)
        return MPSMessage(mps=mps, bond_names=boundary.bond_names)

    def compute_outgoing(
        self,
        sender_block,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, MPSMessage]],
    ) -> MPSMessage:
        N = len(target.bond_names)
        if N == 0:
            return MPSMessage(mps=None, bond_names=())

        # TN2D 경로 (paper-faithful). 다음 조건이면 fallback:
        #   target에 logical bond 포함, 또는 sender에 L_acc 텐서.
        if target.all_spatial and not _has_l_acc(sender_tensors):
            return self._compute_outgoing_2d(sender_block, sender_tensors, target, incoming)
        return self._compute_outgoing_flat(sender_block, sender_tensors, target, incoming)

    @staticmethod
    def _canonicalize_mps(msg: MPSMessage, inc_idx: int):
        """MPS virtual bond → 캐시 키에 stable한 canonical 이름 (`_v_<inc_idx>_<pos>`).

        Returns (inds_list, shape_list, array_list). msg.mps is None이면 빈 리스트들.
        """
        if msg.mps is None:
            return [], [], []
        phys_set = set(msg.bond_names)
        tensors = list(msg.mps.tensors)
        N = len(tensors)
        site_inds = [t.inds for t in tensors]
        # 인접 사이트 간 공통 ind = 그 사이의 virtual bond
        canonical_map: Dict[str, str] = {}
        for i in range(N - 1):
            common = set(site_inds[i]) & set(site_inds[i + 1])
            if len(common) != 1:
                raise ValueError(
                    f"MPS structure unexpected: {len(common)} common inds between sites {i}, {i+1}"
                )
            canonical_map[next(iter(common))] = f"_v_{inc_idx}_{i}"

        new_inds = [tuple(canonical_map.get(ind, ind) for ind in inds) for inds in site_inds]
        shapes = [tuple(t.shape) for t in tensors]
        arrays = [np.asarray(t.data) for t in tensors]
        return new_inds, shapes, arrays

    def _compute_outgoing_flat(
        self,
        sender_block,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, MPSMessage]],
    ) -> MPSMessage:
        # sender 부분 — 구조 고정
        inputs = [tuple(t.inds) for t in sender_tensors]
        shapes = [tuple(t.shape) for t in sender_tensors]
        arrays = [np.asarray(t.data) for t in sender_tensors]
        # incoming MPS — canonical 이름으로 매핑
        inc_meta = []
        for inc_idx, (g, msg) in enumerate(incoming):
            c_inds, c_shapes, c_arrays = self._canonicalize_mps(msg, inc_idx)
            inputs.extend(c_inds)
            shapes.extend(c_shapes)
            arrays.extend(c_arrays)
            inc_meta.append((g.bond_names, tuple(c_inds), tuple(c_shapes)))

        cache_key = (
            sender_block, target.bond_names,
            tuple(inputs[:len(sender_tensors)]),
            tuple(shapes[:len(sender_tensors)]),
            tuple(inc_meta),
        )
        expr = self._expr_cache.get(cache_key)
        if expr is None:
            expr = ctg.array_contract_expression(
                inputs=inputs,
                output=tuple(target.bond_names),
                shapes=shapes,
                optimize='greedy',
            )
            self._expr_cache[cache_key] = expr

        flat = np.ascontiguousarray(expr(*arrays))
        mps = self._build_mps(flat, target.bond_names, max_bond=self.chi)
        return MPSMessage(mps=mps, bond_names=target.bond_names)

    def _compute_outgoing_2d(
        self,
        sender_block,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, MPSMessage]],
    ) -> MPSMessage:
        """Paper-faithful boundary MPS contraction via quimb TN2D."""
        from qec_sim.decoders._bsv_blockbp import side_of_a

        Lx, Ly = _infer_block_dims(sender_tensors)

        # target_side: sender 관점 (xmin/xmax/ymin/ymax). target.bonds[0]에서 판정 —
        # sender_block이 cb.block_a면 side_of_a, block_b면 opposite.
        cb = target.bonds[0]
        if sender_block == cb.block_a:
            target_side = side_of_a(cb)
        else:
            target_side = _OPPOSITE_SIDE[side_of_a(cb)]

        tensors = list(sender_tensors)
        for g, msg in incoming:
            tensors.extend(self.inject(msg, g))

        base = qtn.TensorNetwork(tensors)
        tn2d = qtn.TensorNetwork2D.from_TN(
            base, Lx=Lx, Ly=Ly,
            site_tag_id='I{},{}', x_tag_id='X{}', y_tag_id='Y{}',
        )

        method = f'contract_boundary_from_{_OPPOSITE_SIDE[target_side]}'
        # xmin/xmax는 첫 인자 xrange, ymin/ymax는 첫 인자 yrange.
        # 항상 (0, L-1) 전체 범위 — 한쪽 끝에서 반대쪽 끝까지 모두 흡수.
        if target_side in ('xmin', 'xmax'):
            primary_range = (0, Lx - 1)
        else:
            primary_range = (0, Ly - 1)
        contracted = getattr(tn2d, method)(primary_range, max_bond=self.chi)

        # 남은 텐서들을 contract → target.bond_names 외 다 사라짐
        result = contracted.contract(output_inds=list(target.bond_names))
        flat = np.asarray(result.data) if hasattr(result, 'data') else np.asarray(result)
        mps = self._build_mps(flat, target.bond_names, max_bond=self.chi)
        return MPSMessage(mps=mps, bond_names=target.bond_names)

    def normalize(self, m: MPSMessage) -> MPSMessage:
        # In-place 안전: BP loop에서 normalize에 들어오는 mps는 항상 fresh
        # (compute_outgoing/damp/init_uniform/scale 모두 새 MPS 반환).
        if m.mps is None:
            return m
        m.mps.normalize()
        return m

    def damp(self, new: MPSMessage, old: MPSMessage, eta: float) -> MPSMessage:
        if new.mps is None:
            return new
        # Weighted MPS sum: scale 후 quimb의 MPS add → compress to chi
        a = new.mps * (1 - eta)
        b = old.mps * eta
        summed = a + b
        summed.compress(max_bond=self.chi)  # in-place by default
        return MPSMessage(mps=summed, bond_names=new.bond_names)

    def delta(self, new: MPSMessage, old: MPSMessage) -> float:
        if new.mps is None:
            return 0.0
        diff = new.mps - old.mps
        norm_sq = float(diff @ diff)
        return float(np.sqrt(max(norm_sq, 0.0)))

    def inner(self, m_ab: MPSMessage, m_ba: MPSMessage) -> float:
        if m_ab.mps is None:
            return 1.0
        return float(m_ab.mps @ m_ba.mps)

    def scale(self, m: MPSMessage, factor: float) -> MPSMessage:
        if m.mps is None:
            return m
        new_mps = m.mps * factor
        return MPSMessage(mps=new_mps, bond_names=m.bond_names)

    def inject(self, m: MPSMessage, boundary: BoundaryGroup) -> List[qtn.Tensor]:
        if m.mps is None:
            return []
        # MPS 사이트 텐서들을 그대로. virtual bond는 random unique 이름이라 충돌 없음.
        return list(m.mps.tensors)


# ──────────────────────────────────────────────
# MPSBackend의 TN2D 경로 helper
# ──────────────────────────────────────────────

_OPPOSITE_SIDE = {'xmin': 'xmax', 'xmax': 'xmin', 'ymin': 'ymax', 'ymax': 'ymin'}


def _has_l_acc(tensors: List[qtn.Tensor]) -> bool:
    return any('L_acc' in t.tags for t in tensors)


def _infer_block_dims(tensors: List[qtn.Tensor]) -> Tuple[int, int]:
    """site_tag (I{x},{y}) 들에서 블록의 Lx, Ly 추출. site_tag 없으면 raise."""
    max_x = -1
    max_y = -1
    for t in tensors:
        for tag in t.tags:
            if isinstance(tag, str) and tag.startswith('I') and ',' in tag:
                try:
                    sx, sy = tag[1:].split(',')
                    x, y = int(sx), int(sy)
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                except ValueError:
                    continue
    if max_x < 0 or max_y < 0:
        raise ValueError("sender_tensors에 site_tag (I{x},{y})가 없음 — TN2D wrap 불가능")
    return max_x + 1, max_y + 1
