"""BSV (Bravyi-Suchara-Vargo 2014) 텐서 네트워크 — unrotated d×d surface code의
DQMLD 디코딩을 (2d-1)×(2d-1) PEPS로 구성. 본 모듈은 bit-flip on Z-memory 케이스만 다룸
(depolarizing은 후속 단계).

좌표:
    격자 (x, y) ∈ [0, 2d-2]²
    (x+y) 짝수      → 데이터 qubit (총 d²+(d-1)²개)
    x 짝수, y 홀수  → Z stabilizer (총 d(d-1)개)
    x 홀수, y 짝수  → X stabilizer (bit-flip엔 passive)
    bottom 행 (y=0, x 짝수) → 논리 Z̄ 측정용 데이터 qubit (총 d개)

텐서:
    데이터 qubit: weighted COPY rank=#neighbors (+1 if bottom-row), `T[v,...,v] = (1-p, p)`
    Z stab: parity, target = syndrome 비트
    X stab: all-1 (bit-flip 채널에선 X stab 측정값 결정적이라 무관 — passive)
    L_acc: bottom-row 데이터 qubit 모음의 XOR을 개방 인덱스 'L'로 산출

Contract 결과: 개방 인덱스 'L' 차원 2의 텐서 → unnormalized P(L | s).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import quimb.tensor as qtn

Coord = Tuple[int, int]


# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class BSVLayout:
    d: int
    n: int                          # 격자 크기 = 2d - 1
    data_qubits: Tuple[Coord, ...]
    z_stabs: Tuple[Coord, ...]
    x_stabs: Tuple[Coord, ...]
    logical_data: Tuple[Coord, ...]  # bottom 행 데이터 qubit 모음 (logical Z̄ string)


def bsv_layout(d: int) -> BSVLayout:
    """d만 받아 (2d-1)² 격자에서 사이트 분류. stim/회로 의존성 없음."""
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}")
    n = 2 * d - 1
    data: List[Coord] = []
    z: List[Coord] = []
    x: List[Coord] = []
    for x_ in range(n):
        for y_ in range(n):
            if (x_ + y_) % 2 == 0:
                data.append((x_, y_))
            elif y_ % 2 == 1:  # x_ % 2 == 0
                z.append((x_, y_))
            else:               # x_ % 2 == 1, y_ % 2 == 0
                x.append((x_, y_))
    logical = tuple((2 * i, 0) for i in range(d))
    return BSVLayout(
        d=d,
        n=n,
        data_qubits=tuple(data),
        z_stabs=tuple(z),
        x_stabs=tuple(x),
        logical_data=logical,
    )


def neighbors(coord: Coord, n: int) -> List[Coord]:
    """격자 안에서 NSEW 인접 좌표 (boundary는 자동 truncate)."""
    x, y = coord
    out = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            out.append((nx, ny))
    return out


def bond_name(a: Coord, b: Coord) -> str:
    """두 사이트 사이의 bond 인덱스 이름 — 좌표 순서 무관(canonical)."""
    if a > b:
        a, b = b, a
    return f"b_{a[0]}_{a[1]}__{b[0]}_{b[1]}"


# ──────────────────────────────────────────────
# 텐서 빌더
# ──────────────────────────────────────────────

def _parity_tensor(deg: int, target: int) -> np.ndarray:
    """T[i_1, ..., i_deg] = 1 iff XOR(i_1, ..., i_deg) == target else 0."""
    if deg == 0:
        return np.array(1.0 if target == 0 else 0.0)
    t = np.zeros((2,) * deg)
    for idx in itertools.product([0, 1], repeat=deg):
        if sum(idx) % 2 == target:
            t[idx] = 1.0
    return t


def _copy_tensor(deg: int, p: float) -> np.ndarray:
    """Weighted COPY: T[v,...,v] = (1-p) if v=0 else p; otherwise 0.

    데이터 qubit의 X-error 변수가 인접 모든 bond에 동일하게 전파되는 구조.
    """
    if deg == 0:
        return np.array(1.0)
    t = np.zeros((2,) * deg)
    t[(0,) * deg] = 1.0 - p
    t[(1,) * deg] = p
    return t


# ──────────────────────────────────────────────
# TN 조립
# ──────────────────────────────────────────────

def build_bsv_tn(
    d: int,
    p: float,
    syndrome_at_zstab: Dict[Coord, int],
) -> qtn.TensorNetwork:
    """bit-flip Z-memory용 BSV TN 구성.

    Args:
        d: 코드 distance
        p: 데이터 qubit별 X-error 확률
        syndrome_at_zstab: {(x, y): 0/1} — Z stab 좌표 → syndrome 비트

    Returns:
        quimb TensorNetwork. 개방 인덱스 'L' 차원 2: contract → P(L | s).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    layout = bsv_layout(d)

    # logical_data → 인덱스 0..d-1, bottom 데이터 qubit이 'log_<i>' bond을 추가로 가짐
    log_idx_of: Dict[Coord, int] = {c: i for i, c in enumerate(layout.logical_data)}

    tensors: List[qtn.Tensor] = []

    # 데이터 qubit: weighted COPY (bottom-row이면 logical bond 추가)
    for c in layout.data_qubits:
        nbrs = neighbors(c, layout.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        if c in log_idx_of:
            inds.append(f"log_{log_idx_of[c]}")
        tensors.append(qtn.Tensor(_copy_tensor(len(inds), p), inds, tags=['data', f'D_{c[0]}_{c[1]}']))

    # Z stabilizer: parity tensor with syndrome
    for c in layout.z_stabs:
        nbrs = neighbors(c, layout.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        s = int(syndrome_at_zstab.get(c, 0)) & 1
        tensors.append(qtn.Tensor(_parity_tensor(len(inds), s), inds, tags=['zstab', f'Z_{c[0]}_{c[1]}']))

    # X stabilizer: all-1 (bit-flip엔 passive — 데이터 qubit COPY와 결합 시 무영향)
    for c in layout.x_stabs:
        nbrs = neighbors(c, layout.n)
        inds = [bond_name(c, nb) for nb in nbrs]
        tensors.append(qtn.Tensor(np.ones((2,) * len(inds)), inds, tags=['xstab', f'X_{c[0]}_{c[1]}']))

    # Logical accumulator: bottom-row 데이터 qubit XOR == L
    log_inds = [f"log_{i}" for i in range(d)] + ["L"]
    tensors.append(qtn.Tensor(_parity_tensor(d + 1, 0), log_inds, tags=['logical', 'L_acc']))

    return qtn.TensorNetwork(tensors)


def contract_marginal(tn: qtn.TensorNetwork) -> np.ndarray:
    """TN을 contract하고 'L' 개방 인덱스에 대한 (2,) 벡터 반환 (정규화 안 됨)."""
    result = tn.contract(output_inds=["L"])
    return np.asarray(result.data) if hasattr(result, 'data') else np.asarray(result)
