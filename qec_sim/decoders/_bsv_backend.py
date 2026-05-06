"""BlockBP의 message backend 추상.

paper Eq. (6) BP rule은 메시지 표현(flat 텐서, MPS, density matrix, ...)와 무관함.
이 모듈은 그 추상을 정의 — BP 루프는 `MessageBackend`만 호출하고 구체 표현엔 무지.

현재 구현:
    FlatBackend — 메시지가 (2,)^N flat 텐서. 작은 d / 짧은 boundary에 효율.

후속 구현:
    MPSBackend — 메시지가 quimb MPS, χ 로 SVD truncation. paper 본 알고리즘.
"""
from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable

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

    def init_uniform(self, boundary: BoundaryGroup) -> np.ndarray:
        return np.ones(boundary.shape, dtype=np.float64)

    def compute_outgoing(
        self,
        sender_block,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, np.ndarray]],
    ) -> np.ndarray:
        # FlatBackend는 sender_block 정보 안 씀 (TN2D 활용 안 함).
        del sender_block
        tensors = list(sender_tensors)
        for g, msg in incoming:
            tensors.append(qtn.Tensor(msg, list(g.bond_names), tags=['msg_in']))
        sub = qtn.TensorNetwork(tensors)
        result = sub.contract_compressed(
            'greedy', max_bond=self.max_bond, output_inds=list(target.bond_names),
        )
        return np.asarray(result.data) if hasattr(result, 'data') else np.asarray(result)

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
        return self._compute_outgoing_flat(sender_tensors, target, incoming)

    def _compute_outgoing_flat(
        self,
        sender_tensors: List[qtn.Tensor],
        target: BoundaryGroup,
        incoming: List[Tuple[BoundaryGroup, MPSMessage]],
    ) -> MPSMessage:
        tensors = list(sender_tensors)
        for g, msg in incoming:
            tensors.extend(self.inject(msg, g))
        sub = qtn.TensorNetwork(tensors)
        result = sub.contract_compressed(
            'greedy', max_bond=self.max_bond_block,
            output_inds=list(target.bond_names),
        )
        flat = np.asarray(result.data) if hasattr(result, 'data') else np.asarray(result)
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
        if m.mps is None:
            return m
        new_mps = m.mps.copy()
        new_mps.normalize()
        return MPSMessage(mps=new_mps, bond_names=m.bond_names)

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
