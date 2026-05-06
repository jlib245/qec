"""BSV BlockBP partition (C1) 단위 테스트.

검증:
- 사이트 → 블록 매핑 정확성
- 블록 사이트 합 = (2d-1)²
- cross bond는 양 끝 블록이 다름
- k = 2d-1이면 한 블록 (cross bond 0)
- log bonds: bottom-row 데이터 qubit이 logical_block과 같은 블록이면 internal, 아니면 cross
"""
import pytest

from qec_sim.decoders._bsv_blockbp import partition_bsv_lattice
from qec_sim.decoders._bsv_peps import bsv_layout


@pytest.mark.parametrize("d,k", [(3, 1), (3, 2), (3, 3), (3, 5), (5, 3), (5, 4), (7, 3)])
def test_partition_covers_all_sites(d, k):
    part = partition_bsv_lattice(d, k)
    assert part.n == 2 * d - 1
    total_sites = sum(len(sites) for sites in part.block_sites.values())
    assert total_sites == part.n * part.n


def test_single_block_when_k_covers_lattice():
    """k ≥ 2d-1이면 모든 사이트 한 블록. lattice cross bond 0개."""
    d = 3
    part = partition_bsv_lattice(d, k=2 * d - 1)
    assert part.num_blocks == 1
    assert part.lattice_cross_bonds == []
    # logical_block (0,0)이라서 bottom-row 데이터 qubit 다 (0,0) 블록 → log cross 없음
    assert part.logical_cross_bonds == []


def test_k_one_creates_one_block_per_site():
    d = 3
    part = partition_bsv_lattice(d, k=1)
    n = part.n
    assert part.num_blocks == n * n
    # k=1이면 모든 인접 bond가 cross bond
    expected_bonds = 2 * n * (n - 1)  # 가로 + 세로
    assert len(part.lattice_cross_bonds) == expected_bonds


def test_cross_bonds_are_actually_cross():
    part = partition_bsv_lattice(d=5, k=3)
    for cb in part.lattice_cross_bonds:
        assert cb.block_a != cb.block_b


def test_cross_bond_block_pairs_match_partition():
    """cross bond의 (block_a, block_b)가 site_to_block과 일관."""
    from qec_sim.decoders._bsv_peps import neighbors, bond_name as bn_fn
    part = partition_bsv_lattice(d=5, k=3)
    # bond_name에서 좌표 역추출은 안 하고, 모든 인접 pair 다시 enumerate해서 cross인 것 카운트
    seen = set()
    expected_pairs = []
    for c, blk_a in part.site_to_block.items():
        for nb in neighbors(c, part.n):
            bn = bn_fn(c, nb)
            if bn in seen:
                continue
            seen.add(bn)
            blk_b = part.site_to_block[nb]
            if blk_a != blk_b:
                expected_pairs.append((bn, frozenset([blk_a, blk_b])))
    found_pairs = [(cb.bond_name, frozenset([cb.block_a, cb.block_b])) for cb in part.lattice_cross_bonds]
    assert sorted(expected_pairs) == sorted(found_pairs)


def test_logical_cross_bonds_d3_k3():
    """d=3 k=3, logical_block=(0,0):
    bottom-row 데이터 qubit (0,0), (2,0), (4,0).
    (0,0)→블록(0,0), (2,0)→블록(0,0), (4,0)→블록(1,0).
    → log_0, log_1 internal; log_2 cross.
    """
    part = partition_bsv_lattice(d=3, k=3)
    cross_names = {cb.bond_name for cb in part.logical_cross_bonds}
    assert cross_names == {"log_2"}


def test_logical_cross_bonds_d3_k1():
    """k=1이면 모든 사이트가 단독 블록 → 첫 한 개를 빼고 모든 log_<i>가 cross."""
    part = partition_bsv_lattice(d=3, k=1)
    cross_names = {cb.bond_name for cb in part.logical_cross_bonds}
    # logical_block=(0,0), (0,0) 데이터 qubit이 거기 있음 → log_0 internal
    # log_1, log_2 cross
    assert cross_names == {"log_1", "log_2"}


def test_invalid_params():
    with pytest.raises(ValueError):
        partition_bsv_lattice(d=3, k=0)
    with pytest.raises(ValueError):
        partition_bsv_lattice(d=1, k=2)


def test_partition_is_deterministic():
    p1 = partition_bsv_lattice(d=5, k=3)
    p2 = partition_bsv_lattice(d=5, k=3)
    assert p1.site_to_block == p2.site_to_block
    assert sorted(cb.bond_name for cb in p1.lattice_cross_bonds) == \
           sorted(cb.bond_name for cb in p2.lattice_cross_bonds)
