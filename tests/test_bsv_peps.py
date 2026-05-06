"""BSV PEPS 텐서 네트워크 단위 테스트 (paper Sec III A 재현 — bit-flip Z-mem).

검증 포인트:
- 격자 layout (사이트 카운트, 종류, logical row)
- bond 인덱스 canonical (좌표 순서 무관)
- d=3 brute force와 element-wise 일치 (다중 syndrome)
- 경계 케이스 (p=0, p=0.5)
- d=5 contract 가능 (brute force는 2^41이라 불가능; 동작만 확인)
"""
from __future__ import annotations

import numpy as np
import pytest

from qec_sim.decoders._bsv_peps import (
    bond_name,
    bsv_layout,
    build_bsv_tn,
    contract_marginal,
    neighbors,
)


# ──────────────────────────────────────────────
# 격자 layout
# ──────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 3, 5, 7])
def test_layout_counts(d):
    L = bsv_layout(d)
    assert L.n == 2 * d - 1
    assert len(L.data_qubits) == d * d + (d - 1) * (d - 1)
    assert len(L.z_stabs) == d * (d - 1)
    assert len(L.x_stabs) == d * (d - 1)
    # 합이 (2d-1)²
    assert len(L.data_qubits) + len(L.z_stabs) + len(L.x_stabs) == L.n * L.n
    # logical row: d개
    assert len(L.logical_data) == d
    assert all(y == 0 and x % 2 == 0 for x, y in L.logical_data)


def test_layout_disjoint():
    """data / Z / X stab 사이트 집합은 disjoint."""
    L = bsv_layout(5)
    s_data = set(L.data_qubits)
    s_z = set(L.z_stabs)
    s_x = set(L.x_stabs)
    assert s_data.isdisjoint(s_z)
    assert s_data.isdisjoint(s_x)
    assert s_z.isdisjoint(s_x)


def test_layout_invalid_d():
    with pytest.raises(ValueError):
        bsv_layout(1)


def test_neighbors_within_grid():
    n = 5
    for x in range(n):
        for y in range(n):
            for nx, ny in neighbors((x, y), n):
                assert 0 <= nx < n and 0 <= ny < n
                # NSEW only — 거리 1
                assert abs(nx - x) + abs(ny - y) == 1


def test_neighbors_count():
    """모서리 2, 변 3, 내부 4."""
    n = 5
    assert len(neighbors((0, 0), n)) == 2
    assert len(neighbors((0, 2), n)) == 3
    assert len(neighbors((2, 2), n)) == 4
    assert len(neighbors((4, 4), n)) == 2


# ──────────────────────────────────────────────
# Bond 인덱스 canonical
# ──────────────────────────────────────────────

def test_bond_name_canonical():
    # 좌표 순서 무관
    assert bond_name((0, 1), (0, 2)) == bond_name((0, 2), (0, 1))
    assert bond_name((1, 1), (2, 1)) == bond_name((2, 1), (1, 1))


def test_bond_name_distinct_for_different_pairs():
    names = {bond_name((0, 1), (0, 2)), bond_name((1, 1), (2, 1)), bond_name((2, 2), (3, 2))}
    assert len(names) == 3


# ──────────────────────────────────────────────
# Brute force 일치 (d=3)
# ──────────────────────────────────────────────

def _brute_force_marginal(d: int, p: float, syndrome_dict: dict) -> np.ndarray:
    """모든 2^N 에러 패턴을 enumerate해서 P(L|s) 계산. d=3 (N=13)에서만 실용적."""
    L = bsv_layout(d)
    dq_list = list(L.data_qubits)
    dq_idx = {c: i for i, c in enumerate(dq_list)}

    def zstab_data_neighbors(c):
        return [dq_idx[nb] for nb in neighbors(c, L.n) if (nb[0] + nb[1]) % 2 == 0]

    z_nbrs = {c: zstab_data_neighbors(c) for c in L.z_stabs}
    log_idx = [dq_idx[c] for c in L.logical_data]

    target = {c: int(syndrome_dict.get(c, 0)) & 1 for c in L.z_stabs}

    out = np.zeros(2)
    N = len(dq_list)
    for bits in range(2 ** N):
        e = [(bits >> i) & 1 for i in range(N)]
        syn = {c: sum(e[i] for i in z_nbrs[c]) % 2 for c in L.z_stabs}
        if syn != target:
            continue
        weight = (p ** sum(e)) * ((1 - p) ** (N - sum(e)))
        log_bit = sum(e[i] for i in log_idx) % 2
        out[log_bit] += weight
    return out


@pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.15])
@pytest.mark.parametrize("syn_label,syndrome_dict", [
    ("zero", {}),
    ("one_corner", {(0, 1): 1}),
    ("two_adjacent", {(0, 1): 1, (2, 1): 1}),
    ("opposite_corners", {(0, 1): 1, (4, 3): 1}),
])
def test_bsv_tn_matches_brute_force_d3(p, syn_label, syndrome_dict):
    bf = _brute_force_marginal(d=3, p=p, syndrome_dict=syndrome_dict)
    tn = build_bsv_tn(d=3, p=p, syndrome_at_zstab=syndrome_dict)
    bsv = contract_marginal(tn)

    bf_norm = bf / bf.sum()
    bsv_norm = bsv / bsv.sum()
    np.testing.assert_allclose(bf_norm, bsv_norm, atol=1e-12)


# ──────────────────────────────────────────────
# 경계 케이스
# ──────────────────────────────────────────────

def test_zero_noise_zero_syndrome_no_logical_flip():
    """p=0 & 모든 syndrome 0이면 P(L=0|s) = 1, P(L=1|s) = 0 정확."""
    tn = build_bsv_tn(d=3, p=0.0, syndrome_at_zstab={})
    pL = contract_marginal(tn)
    pL_norm = pL / pL.sum()
    np.testing.assert_allclose(pL_norm, [1.0, 0.0], atol=1e-12)


def test_marginal_finite_and_nonneg():
    """랜덤 syndrome에서도 결과는 finite, non-negative, sum>0."""
    rng = np.random.default_rng(0)
    L = bsv_layout(3)
    for _ in range(20):
        syn = {c: int(rng.integers(2)) for c in L.z_stabs}
        tn = build_bsv_tn(d=3, p=0.05, syndrome_at_zstab=syn)
        pL = contract_marginal(tn)
        assert pL.shape == (2,)
        assert np.isfinite(pL).all()
        assert (pL >= 0).all()
        assert pL.sum() > 0


# ──────────────────────────────────────────────
# 더 큰 d에서 contract 동작 (brute force 불가능)
# ──────────────────────────────────────────────

def test_contract_d5_no_crash():
    L = bsv_layout(5)
    # 적당한 임의 syndrome: 10개 켜기
    rng = np.random.default_rng(1)
    syn = {c: int(rng.integers(2)) for c in L.z_stabs}
    tn = build_bsv_tn(d=5, p=0.05, syndrome_at_zstab=syn)
    pL = contract_marginal(tn)
    assert pL.shape == (2,)
    assert np.isfinite(pL).all()
    assert pL.sum() > 0
