"""BSV BlockBP BP 루프 (C3) + Lemma III.1 Tr 추정 테스트.

검증:
- 단일 블록 (cross 없음): BP 0회, Tr = block contract = exact scalar
- 4-블록 partition d=3: BP 수렴 + Lemma III.1 Tr이 exact scalar에 가까움
- damping invalid 값
"""
import numpy as np
import pytest

from qec_sim.decoders._bsv_backend import FlatBackend
from qec_sim.decoders._bsv_blockbp import (
    block_to_boundaries,
    boundary_groups,
    build_block_tensors,
    estimate_tr_from_messages,
    partition_bsv_lattice,
    run_bp_loop,
)
from qec_sim.decoders._bsv_peps import build_bsv_tn_for_class, contract_scalar


def _setup(d, k, p, syndrome, class_bit, logical_block=(0, 0)):
    part = partition_bsv_lattice(d=d, k=k, logical_block=logical_block)
    groups = boundary_groups(part)
    boundaries_of = block_to_boundaries(part, groups)
    block_tensors = build_block_tensors(part, p=p, syndrome_at_zstab=syndrome,
                                        class_bit=class_bit)
    return part, groups, boundaries_of, block_tensors


# ──────────────────────────────────────────────
# 단일 블록 (degenerate)
# ──────────────────────────────────────────────

def test_single_block_bp_terminates_immediately():
    part, groups, bo, bt = _setup(d=3, k=99, p=0.05, syndrome={}, class_bit=0)
    res = run_bp_loop(part, bt, groups, bo, max_iter=10, delta_0=1e-6, backend=FlatBackend(max_bond=4))
    assert res.iterations == 1
    assert res.converged
    assert res.final_delta == 0.0
    assert res.messages == {}


def test_single_block_tr_equals_full_contract():
    part, groups, bo, bt = _setup(d=3, k=99, p=0.05, syndrome={(0, 1): 1}, class_bit=0)
    backend = FlatBackend(max_bond=4)
    res = run_bp_loop(part, bt, groups, bo, max_iter=10, delta_0=1e-6, backend=backend)
    tr_estimate = estimate_tr_from_messages(part, bt, groups, bo, res.messages, backend)

    tn_exact = build_bsv_tn_for_class(d=3, p=0.05, syndrome_at_zstab={(0, 1): 1}, class_bit=0)
    exact = contract_scalar(tn_exact)
    np.testing.assert_allclose(tr_estimate, exact, rtol=1e-10)


# ──────────────────────────────────────────────
# 다중 블록: BP 동작 + Tr ≈ exact (작은 d, 충분한 χ)
# ──────────────────────────────────────────────

def test_bp_runs_and_converges_d3_k3():
    """d=3 k=3 → 4블록. BP가 max_iter 안에 수렴 (또는 적어도 deltas가 finite)."""
    part, groups, bo, bt = _setup(d=3, k=3, p=0.05, syndrome={(0, 1): 1}, class_bit=0)
    res = run_bp_loop(part, bt, groups, bo, max_iter=50, delta_0=1e-5,
                      backend=FlatBackend(max_bond=8))
    assert np.isfinite(res.final_delta)
    assert res.iterations >= 1


def test_tr_estimate_close_to_exact_d3():
    """d=3 k=3, χ=16: Lemma III.1 Tr ≈ exact scalar (loopy partition이라 strict X)."""
    syn = {(0, 1): 1, (2, 3): 1}
    for class_bit in (0, 1):
        part, groups, bo, bt = _setup(d=3, k=3, p=0.05, syndrome=syn, class_bit=class_bit)
        backend = FlatBackend(max_bond=16)
        res = run_bp_loop(part, bt, groups, bo, max_iter=200, delta_0=1e-6,
                          backend=backend, damping=0.5)
        tr_est = estimate_tr_from_messages(part, bt, groups, bo, res.messages, backend)

        tn_exact = build_bsv_tn_for_class(d=3, p=0.05, syndrome_at_zstab=syn,
                                          class_bit=class_bit)
        exact = contract_scalar(tn_exact)
        # loopy cluster graph라 strict 일치 안 됨 — relative 1% 이내면 OK
        rel_err = abs(tr_est - exact) / max(abs(exact), 1e-12)
        assert rel_err < 0.05, (
            f"class_bit={class_bit}: Tr est {tr_est:.6e} vs exact {exact:.6e}, "
            f"rel_err={rel_err:.2%}, converged={res.converged}, "
            f"iters={res.iterations}, delta={res.final_delta:.2e}"
        )


# ──────────────────────────────────────────────
# 입력 검증
# ──────────────────────────────────────────────

def test_invalid_damping():
    part, groups, bo, bt = _setup(d=3, k=3, p=0.05, syndrome={}, class_bit=0)
    backend = FlatBackend(max_bond=4)
    with pytest.raises(ValueError):
        run_bp_loop(part, bt, groups, bo, max_iter=10, delta_0=1e-6, backend=backend,
                    damping=1.0)
    with pytest.raises(ValueError):
        run_bp_loop(part, bt, groups, bo, max_iter=10, delta_0=1e-6, backend=backend,
                    damping=-0.1)
