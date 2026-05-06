"""Paper-faithful BlockBP for BSV PEPS, on quimb's BeliefPropagationCommon.

paper Algorithm 1을 자체 BP 루프로 구현. quimb의 인프라를 부분 활용:
    - `BeliefPropagationCommon`: run() 루프, damping, DIIS, 수렴 통계
    - `create_lazy_community_edge_map`: site_tags 기반 partition + cross bond 자동 검출
    - `combine_local_contractions`: Bethe free entropy 누적 시 underflow 방지

자체 작성:
    - iterate(): paper Algorithm 1 메시지 업데이트 (sequential/parallel)
    - contract(): Bethe 추정값 (block + edge overlap)
    - _compute_outgoing(): block + incoming 메시지 → outgoing 메시지

메시지 표현 — `max_chi`로 dispatch:
    - max_chi is None: dense 메시지 (numpy array). vanilla `contract_l1bp` 결과와 일치.
    - max_chi: int (≥1): MPS 메시지 (`qtn.MatrixProductState`), bond dim ≤ max_chi.
        outgoing 계산 시 dense intermediate → `from_dense(max_bond=chi)` 압축. paper의 메시지
        chi-truncation 구현 (boundary MPS sweep은 step 3b 가서 추가).

검증 전략:
    - dense (max_chi=None): vanilla contract_l1bp와 numerical roundoff 수준 일치
    - MPS 큰 χ: dense 결과로 수렴 — chi 충분히 크면 truncation no-op
    - MPS chi sweep: chi↑ → exact contraction에 단조 수렴
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import quimb.tensor as qtn
from quimb.tensor.belief_propagation.bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)
from quimb.utils import oset


# ──────────────────────────────────────────────
# MPS-aware normalize/distance/damping (free functions, BPC에 callable로 주입)
# ──────────────────────────────────────────────

def _mps_normalize(mps: qtn.MatrixProductState) -> qtn.MatrixProductState:
    """In-place L2 normalize of MPS, return same object."""
    mps.normalize()
    return mps


def _mps_distance(a: qtn.MatrixProductState, b: qtn.MatrixProductState) -> float:
    """L2 distance ||A - B||_2 between two MPSes."""
    diff = a - b
    nrm_sq = float(diff @ diff)
    return float(np.sqrt(max(nrm_sq, 0.0)))


def _make_mps_damping_fn(eta: float, max_chi: int):
    """Closure: weighted MPS sum + compress to max_chi."""
    if not (0.0 <= eta < 1.0):
        raise ValueError(f"damping must be in [0, 1), got {eta}")

    def _fn(old, new):
        if eta == 0.0:
            return new
        # bond dim 합쳐짐 → compress로 chi 이내로 자름
        s = (1 - eta) * new + eta * old
        s.compress(max_bond=max_chi)
        return s
    return _fn


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class BlockBPMps(BeliefPropagationCommon):
    """BlockBP with dense or MPS messages, on top of BeliefPropagationCommon.

    Args:
        tn: closed TensorNetwork to contract via BP.
        site_tags: 블록 분할. 각 텐서가 정확히 하나의 site_tag를 가져야.
            None이면 tn.site_tags 자동.
        max_chi:
            None — dense 메시지 (numpy array). step 2 검증 모드.
            int ≥ 1 — MPS 메시지, bond dim ≤ max_chi. step 3 paper-faithful 모드.
        damping: float in [0, 1) — `(1-eta)*new + eta*old`. dense/MPS 모두 동일 의미.
        update: 'sequential' | 'parallel'.
        normalize, distance: 보통 None (자동: MPS 모드면 MPS-aware, dense면 BPC default L2).
        local_convergence: 모든 incoming이 수렴한 노드는 update skip (BPC 패턴).
        optimize: cotengra path optimizer.
    """

    def __init__(
        self,
        tn,
        site_tags=None,
        *,
        max_chi: Optional[int] = None,
        damping: float = 0.0,
        update: str = "sequential",
        normalize=None,
        distance=None,
        local_convergence: bool = True,
        optimize: str = "auto-hq",
        contract_every=None,
        inplace: bool = False,
        **contract_opts,
    ):
        if max_chi is not None and max_chi < 1:
            raise ValueError(f"max_chi must be >= 1, got {max_chi}")

        # MPS 모드면 normalize/distance/damping을 MPS-aware callable로 주입
        # (BPC default는 numpy array 가정 — MPS 객체엔 안 맞음)
        if max_chi is not None:
            if normalize is None:
                normalize = _mps_normalize
            if distance is None:
                distance = _mps_distance
            if not callable(damping):
                damping = _make_mps_damping_fn(float(damping), max_chi)

        super().__init__(
            tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )
        self.max_chi = max_chi
        self.local_convergence = local_convergence
        self.optimize = optimize
        self.contract_opts = contract_opts

        if site_tags is None:
            self.site_tags = tuple(self.tn.site_tags)
        else:
            self.site_tags = tuple(site_tags)

        (
            self.edges,
            self.neighbors,
            self.local_tns,
            self.touch_map,
        ) = create_lazy_community_edge_map(self.tn, self.site_tags)

        self.touched: oset = oset()

        # 초기 메시지: local_tn[i] dense contract → bix 위 텐서, 그 다음 mode별 변환
        self.messages: Dict[Tuple[str, str], object] = {}
        for pair, bix in self.edges.items():
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i]
                tm = tn_i.contract(
                    all,
                    output_inds=bix,
                    optimize=self.optimize,
                    drop_tags=True,
                    **self.contract_opts,
                )
                arr = np.asarray(tm.data)
                if max_chi is None:
                    msg = self._normalize_fn(arr)
                else:
                    msg = self._dense_to_mps(arr, bix)
                    msg = self._normalize_fn(msg)
                self.messages[(i, j)] = msg

    # ─────────── helpers: dense ↔ MPS ───────────

    def _dense_to_mps(self, arr: np.ndarray, bix: Tuple[str, ...]) -> qtn.MatrixProductState:
        """dense (2,)^N tensor → MPS with phys legs reindex'd to bix, bond dim ≤ max_chi."""
        N = len(bix)
        if N == 0:
            raise ValueError("empty boundary — degenerate case not yet handled")
        mps = qtn.MatrixProductState.from_dense(
            arr, dims=(2,) * N, max_bond=self.max_chi,
        )
        # MPS phys legs는 from_dense이 'k0', 'k1', ...로 만듦 → bix로 reindex
        mps.reindex({f"k{i}": bix[i] for i in range(N)}, inplace=True)
        return mps

    def _msg_as_tensors(self, msg, bix: Tuple[str, ...]) -> List[qtn.Tensor]:
        """메시지를 sub-TN에 inject할 텐서 리스트로 변환.

        dense 모드: numpy array → 단일 qtn.Tensor (bix indexing)
        MPS 모드: 메시지의 site tensor list 그대로
        """
        if isinstance(msg, np.ndarray):
            return [qtn.Tensor(msg, list(bix), tags=["msg_in"])]
        # MatrixProductState — site tensors 그대로 (virtual bond는 unique 이름이라 충돌 없음)
        return list(msg.tensors)

    # ─────────── 메시지 업데이트 ───────────

    def _compute_outgoing(self, i, j):
        """block i + (j 제외 incoming 메시지들) → outgoing 메시지.

        반환은 mode 의존: dense면 numpy array, MPS면 MatrixProductState.
        둘 다 _normalize_fn 적용된 상태.
        """
        bix = self.edges[(i, j) if i < j else (j, i)]
        local_tn = self.local_tns[i]

        msg_tensors: List[qtn.Tensor] = []
        for k in self.neighbors[i]:
            if k == j:
                continue
            ki_bix = self.edges[(k, i) if k < i else (i, k)]
            msg_tensors.extend(self._msg_as_tensors(self.messages[(k, i)], ki_bix))

        sub = qtn.TensorNetwork((local_tn, *msg_tensors), virtual=False)
        result = sub.contract(
            all,
            output_inds=bix,
            optimize=self.optimize,
            **self.contract_opts,
        )
        arr = np.asarray(result.data)

        if self.max_chi is None:
            return self._normalize_fn(arr)
        mps = self._dense_to_mps(arr, bix)
        return self._normalize_fn(mps)

    def iterate(self, tol: float = 5e-6):
        """paper Algorithm 1: 모든 (i→j) 메시지 한 라운드 업데이트.

        sequential: 새 메시지 즉시 다음 update에 반영
        parallel: 같은 라운드 내에서는 이전 메시지만 사용
        """
        if (not self.local_convergence) or (not self.touched):
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched: oset = oset()

        def _update_msg(key, new_msg):
            nonlocal nconv, max_mdiff
            old_msg = self.messages[key]
            mdiff = self._distance_fn(new_msg, old_msg)
            if self.damping:
                new_msg = self._damping_fn(old_msg, new_msg)
            if mdiff > tol:
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1
            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = new_msg

        if self.update == "parallel":
            new_data = {}
            while self.touched:
                key = self.touched.pop()
                new_data[key] = self._compute_outgoing(*key)
            for key, msg in new_data.items():
                _update_msg(key, msg)

        elif self.update == "sequential":
            while self.touched:
                key = self.touched.pop()
                msg = self._compute_outgoing(*key)
                _update_msg(key, msg)

        else:
            raise ValueError(f"unknown update mode: {self.update}")

        self.touched = new_touched
        return {"nconv": nconv, "ncheck": ncheck, "max_mdiff": max_mdiff}

    # ─────────── 최종 contract (Bethe free entropy 추정) ───────────

    def contract(self, strip_exponent: bool = False, check_zero: bool = True):
        """ZBethe = ∏_v Tr(T_v ∏ m_{u→v}) / ∏_{(u,v)} <m_{u→v}|m_{v→u}>.

        block scalar는 +1 power, edge overlap은 -1 power로 누적.
        """
        zvals = []

        # block scalar: local_tn + 모든 incoming 메시지 → scalar
        for site, tn_i in self.local_tns.items():
            if site in self.neighbors:
                msg_tensors: List[qtn.Tensor] = []
                for k in self.neighbors[site]:
                    ki_bix = self.edges[(k, site) if k < site else (site, k)]
                    msg_tensors.extend(self._msg_as_tensors(self.messages[(k, site)], ki_bix))
                tval = qtn.tensor_contract(
                    *tn_i,
                    *msg_tensors,
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            else:
                tval = tn_i.contract(
                    all,
                    output_inds=(),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            zvals.append((tval, 1))

        # edge overlap: <m_{i→j} | m_{j→i}>
        for i, j in self.edges:
            ij_bix = self.edges[(i, j)]
            ma_tensors = self._msg_as_tensors(self.messages[(i, j)], ij_bix)
            mb_tensors = self._msg_as_tensors(self.messages[(j, i)], ij_bix)
            mval = qtn.tensor_contract(
                *ma_tensors,
                *mb_tensors,
                optimize=self.optimize,
                **self.contract_opts,
            )
            zvals.append((mval, -1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────

def contract_blockbp_mps(
    tn,
    *,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    site_tags=None,
    max_chi: Optional[int] = None,
    damping: float = 0.0,
    update: str = "sequential",
    diis: bool = False,
    local_convergence: bool = True,
    optimize: str = "auto-hq",
    strip_exponent: bool = False,
    info=None,
    progbar: bool = False,
    **contract_opts,
):
    """BlockBPMps 만들고 run + contract 한 번에 — vanilla `contract_l1bp` 시그니처와 호환."""
    bp = BlockBPMps(
        tn,
        site_tags=site_tags,
        max_chi=max_chi,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        optimize=optimize,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        info=info,
        progbar=progbar,
    )
    return bp.contract(strip_exponent=strip_exponent)
