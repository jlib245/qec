"""BlockBPMps의 n_workers thread pool이 직렬과 수치적으로 동일한지 회귀 검증.

`update="parallel"` 모드에서 n_workers는 sender별 _compute_outgoing을 동시 실행할 뿐
수학적 의미를 바꾸지 않아야 함 — 결과 (Z value, 수렴 통계)가 직렬과 일치해야.
"""
import numpy as np
import pytest

from qec_sim.decoders._blockbp_mps import contract_blockbp_mps
from qec_sim.decoders._bsv_blockbp_mps import add_block_tags_bsv
from qec_sim.decoders._bsv_peps import build_bsv_tn_for_class


def _build_tagged_tn(d: int, p: float, k: int, syndrome, class_bit: int):
    tn = build_bsv_tn_for_class(d=d, p=p, syndrome_at_zstab=syndrome, class_bit=class_bit)
    tn_tagged, site_tags = add_block_tags_bsv(tn, k=k)
    return tn_tagged, site_tags


@pytest.mark.parametrize("max_chi", [None, 4])
def test_parallel_n_workers_matches_serial(max_chi):
    """update='parallel'에서 n_workers=None vs n_workers=4 결과 동일.

    threading은 알고리즘 의미를 바꾸지 않아야 함 — 같은 메시지 set 위에서 같은 update.
    """
    d, k, p = 3, 2, 0.05
    syndrome = {(0, 1): 1}

    z_serial_per_class = []
    z_parallel_per_class = []
    for cb in (0, 1):
        tn, site_tags = _build_tagged_tn(d, p, k, syndrome, cb)

        info1, info2 = {}, {}
        z1 = contract_blockbp_mps(
            tn, site_tags=site_tags,
            max_iterations=50, tol=1e-8, damping=0.3,
            max_chi=max_chi, update="parallel", optimize="auto-hq",
            info=info1, n_workers=None,
        )
        z2 = contract_blockbp_mps(
            tn, site_tags=site_tags,
            max_iterations=50, tol=1e-8, damping=0.3,
            max_chi=max_chi, update="parallel", optimize="auto-hq",
            info=info2, n_workers=4,
        )
        z_serial_per_class.append(float(z1))
        z_parallel_per_class.append(float(z2))

        # 수렴 통계도 일치해야 함 (deterministic update + 같은 init)
        assert info1.get("converged") == info2.get("converged")
        assert info1.get("iterations") == info2.get("iterations")

    np.testing.assert_allclose(
        z_serial_per_class, z_parallel_per_class, rtol=1e-10, atol=1e-12,
        err_msg="n_workers thread pool이 직렬과 수치적 차이를 만들면 안 됨",
    )


def test_n_workers_invalid():
    from qec_sim.decoders._blockbp_mps import BlockBPMps
    tn, site_tags = _build_tagged_tn(d=3, p=0.05, k=3, syndrome={}, class_bit=0)
    with pytest.raises(ValueError):
        BlockBPMps(tn, site_tags=site_tags, n_workers=0)
