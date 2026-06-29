"""MPSBackend (paper-faithful 메시지 표현) 테스트.

검증:
- 잘못된 chi → ValueError
- init_uniform이 올바른 boundary bond_names로 MPS 생성
- normalize / scale 후 inner 결과 일관성
- 작은 d에서 MPSBackend == FlatBackend (큰 chi에선 numerically 동일해야)
"""
import numpy as np
import pytest

from qec_sim.decoders._bsv_backend import FlatBackend, MPSBackend, MPSMessage
from qec_sim.decoders._bsv_blockbp import (
    boundary_groups,
    decode_blockbp,
    partition_bsv_lattice,
)


def test_invalid_chi():
    with pytest.raises(ValueError):
        MPSBackend(chi=0)


def test_init_uniform_has_correct_bond_names():
    part = partition_bsv_lattice(d=3, k=3)
    backend = MPSBackend(chi=4)
    for g in boundary_groups(part):
        m = backend.init_uniform(g)
        assert isinstance(m, MPSMessage)
        # MPS의 physical 인덱스가 bond_names와 일치 (다른 inds는 virtual)
        all_inds = set()
        for t in m.mps.tensors:
            all_inds.update(t.inds)
        for bn in g.bond_names:
            assert bn in all_inds


def test_normalize_gives_unit_norm():
    part = partition_bsv_lattice(d=3, k=3)
    backend = MPSBackend(chi=4)
    g = boundary_groups(part)[0]
    m = backend.init_uniform(g)
    m_n = backend.normalize(m)
    # ⟨n|n⟩ = 1
    assert abs(backend.inner(m_n, m_n) - 1.0) < 1e-10


def test_scale_changes_inner_quadratically():
    """scale(m, c) → inner = c² * inner(m, m)"""
    part = partition_bsv_lattice(d=3, k=3)
    backend = MPSBackend(chi=4)
    g = boundary_groups(part)[0]
    m = backend.normalize(backend.init_uniform(g))
    base = backend.inner(m, m)  # = 1
    m2 = backend.scale(m, 3.0)
    scaled = backend.inner(m2, m2)
    np.testing.assert_allclose(scaled, base * 9.0, rtol=1e-10)


def test_mps_matches_flat_on_d3_decode():
    """충분히 큰 chi에서 MPSBackend 결과가 FlatBackend와 numerically 동일.

    d=3은 2^N ≤ 32 정도라 chi=16이면 MPS truncation이 거의 무손실 — 같은 답이 나와야.
    """
    syn = {(0, 1): 1, (2, 3): 1}
    common = dict(d=3, p=0.07, syndrome_at_zstab=syn, k=3,
                  max_iter=100, delta_0=1e-6, delta_1=1e-2, damping=0.5)
    flat_res = decode_blockbp(**common, backend=FlatBackend(max_bond=16))
    mps_res = decode_blockbp(**common, backend=MPSBackend(chi=16))

    assert flat_res.predicted_class == mps_res.predicted_class
    for cf, cm in zip(flat_res.per_class, mps_res.per_class):
        assert cf.tr_estimate is not None and cm.tr_estimate is not None
        np.testing.assert_allclose(cf.tr_estimate, cm.tr_estimate, rtol=1e-8)


@pytest.mark.parametrize("syndrome", [
    {},
    {(0, 1): 1},
    {(0, 1): 1, (2, 3): 1},
    {(2, 1): 1, (2, 3): 1, (4, 1): 1},
])
def test_mps_argmax_matches_flat_d3(syndrome):
    common = dict(d=3, p=0.07, syndrome_at_zstab=syndrome, k=3,
                  max_iter=100, delta_0=1e-6, delta_1=1e-2, damping=0.5)
    flat_res = decode_blockbp(**common, backend=FlatBackend(max_bond=16))
    mps_res = decode_blockbp(**common, backend=MPSBackend(chi=16))
    assert flat_res.predicted_class == mps_res.predicted_class
