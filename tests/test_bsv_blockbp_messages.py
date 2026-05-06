"""BSV BlockBP 단일 메시지 패스 (C2a — FlatBackend) 테스트.

검증:
- BoundaryGroup 카운트 / 모든 cross bond 정확히 한 그룹에 속함
- block_tensors: 블록별 텐서 합 = 전체 BSV TN 텐서 수 (= n² + 1 (L_acc))
- FlatBackend.compute_outgoing: shape correctness, finite, non-negative
"""
import numpy as np
import pytest

from qec_sim.decoders._bsv_backend import FlatBackend
from qec_sim.decoders._bsv_blockbp import (
    block_to_boundaries,
    boundary_groups,
    build_block_tensors,
    partition_bsv_lattice,
)


# ──────────────────────────────────────────────
# BoundaryGroup
# ──────────────────────────────────────────────

def test_boundary_groups_cover_all_cross_bonds():
    part = partition_bsv_lattice(d=5, k=3)
    groups = boundary_groups(part)
    all_grouped = set()
    for g in groups:
        all_grouped.update(g.bond_names)

    expected = {cb.bond_name for cb in
                list(part.lattice_cross_bonds) + list(part.logical_cross_bonds)}
    assert all_grouped == expected


def test_boundary_groups_canonical_ordering():
    """block_a < block_b 보장."""
    part = partition_bsv_lattice(d=5, k=3)
    for g in boundary_groups(part):
        assert g.block_a < g.block_b


def test_boundary_groups_empty_when_single_block():
    part = partition_bsv_lattice(d=3, k=99)  # k > 2d-1 → 단일 블록
    assert boundary_groups(part) == []


# ──────────────────────────────────────────────
# build_block_tensors
# ──────────────────────────────────────────────

@pytest.mark.parametrize("d,k", [(3, 1), (3, 2), (3, 3), (3, 5), (5, 3)])
def test_block_tensors_total_count(d, k):
    """블록별 텐서 합 = lattice 사이트 수 + 1 (L_acc)."""
    part = partition_bsv_lattice(d, k)
    tn_blocks = build_block_tensors(part, p=0.05, syndrome_at_zstab={}, class_bit=0)
    total = sum(len(ts) for ts in tn_blocks.values())
    assert total == part.n * part.n + 1  # 사이트 (n²) + L_acc


def test_block_tensors_l_acc_in_logical_block():
    part = partition_bsv_lattice(d=3, k=2, logical_block=(1, 1))
    tn_blocks = build_block_tensors(part, p=0.05, syndrome_at_zstab={}, class_bit=0)
    # L_acc 태그는 logical_block에만 있어야
    for blk, ts in tn_blocks.items():
        has_l_acc = any('L_acc' in t.tags for t in ts)
        if blk == part.logical_block:
            assert has_l_acc, f"L_acc not in logical block {blk}"
        else:
            assert not has_l_acc, f"L_acc in non-logical block {blk}"


def test_block_tensors_invalid_class_bit():
    part = partition_bsv_lattice(d=3, k=2)
    with pytest.raises(ValueError):
        build_block_tensors(part, p=0.05, syndrome_at_zstab={}, class_bit=2)


# ──────────────────────────────────────────────
# FlatBackend (메시지 ops)
# ──────────────────────────────────────────────

def test_flat_backend_init_uniform_shape():
    part = partition_bsv_lattice(d=3, k=3)
    groups = boundary_groups(part)
    backend = FlatBackend(max_bond=4)
    for g in groups:
        m = backend.init_uniform(g)
        assert m.shape == g.shape
        assert (m == 1).all()


def test_flat_backend_compute_outgoing_shape():
    part = partition_bsv_lattice(d=3, k=3)
    groups = boundary_groups(part)
    boundaries_of = block_to_boundaries(part, groups)
    block_tensors = build_block_tensors(part, p=0.05, syndrome_at_zstab={}, class_bit=0)
    backend = FlatBackend(max_bond=8)

    g = groups[0]
    incoming = [(og, backend.init_uniform(og))
                for og in boundaries_of[g.block_a] if og is not g]
    out = backend.compute_outgoing(
        sender_block=g.block_a, sender_tensors=block_tensors[g.block_a],
        target=g, incoming=incoming,
    )
    assert out.shape == g.shape
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_flat_backend_invalid_max_bond():
    with pytest.raises(ValueError):
        FlatBackend(max_bond=0)
