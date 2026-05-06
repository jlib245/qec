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

import cotengra as ctg
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
        expr_cache: Optional[Dict[Tuple, object]] = None,
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

        # cotengra contraction expression cache for _compute_outgoing.
        # 키: (i, j, bix, sender_inds, sender_shapes, incoming_meta).
        # 키가 syndrome 데이터와 무관 (구조만 의존)이라 cross-shot 공유 가능.
        # `expr_cache=None`이면 인스턴스 전용 (cross-iteration만 reuse), dict 주면 caller가
        # 여러 BlockBPMps 인스턴스 사이 공유 가능 (cross-shot reuse).
        self._expr_cache: Dict[Tuple, object] = expr_cache if expr_cache is not None else {}

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
        """메시지를 sub-TN에 inject할 텐서 리스트로 변환 (contract() 사용).

        dense 모드: numpy array → 단일 qtn.Tensor (bix indexing)
        MPS 모드: 메시지의 site tensor list 그대로
        """
        if isinstance(msg, np.ndarray):
            return [qtn.Tensor(msg, list(bix), tags=["msg_in"])]
        # MatrixProductState — site tensors 그대로 (virtual bond는 unique 이름이라 충돌 없음)
        return list(msg.tensors)

    def _msg_as_canonical_inputs(self, msg, bix: Tuple[str, ...], inc_idx: int):
        """메시지를 cotengra cache 친화적 (inputs, shapes, arrays) 튜플로.

        cache 키 stable하려면 모든 인덱스 이름이 deterministic해야 함. dense는 이미
        ok (bix), MPS는 virtual bond가 random uuid라 `_v_<inc_idx>_<pos>`로 매핑.
        """
        if isinstance(msg, np.ndarray):
            return [tuple(bix)], [tuple(msg.shape)], [msg]
        # MatrixProductState
        site_inds_list = [t.inds for t in msg.tensors]
        N = len(site_inds_list)
        # 인접 사이트 공통 ind = 그 사이의 virtual bond
        canonical: Dict[str, str] = {}
        for s in range(N - 1):
            common = set(site_inds_list[s]) & set(site_inds_list[s + 1])
            if len(common) != 1:
                raise ValueError(
                    f"MPS structure unexpected: {len(common)} common inds between sites {s}, {s+1}"
                )
            canonical[next(iter(common))] = f"_v_{inc_idx}_{s}"
        # phys leg는 이미 bix로 reindex됨 (canonical 이름)
        new_inds = [
            tuple(canonical.get(ind, ind) for ind in inds) for inds in site_inds_list
        ]
        shapes = [tuple(t.shape) for t in msg.tensors]
        arrays = [np.asarray(t.data) for t in msg.tensors]
        return new_inds, shapes, arrays

    # ─────────── 메시지 업데이트 ───────────

    def _compute_outgoing(self, i, j):
        """block i + (j 제외 incoming 메시지들) → outgoing 메시지.

        반환은 mode 의존: dense면 numpy array, MPS면 MatrixProductState.
        둘 다 _normalize_fn 적용된 상태.

        cotengra path 캐싱: 첫 호출에 expression 빌드, BP warmup 후 cache hit으로 path
        재계산 제거. dense는 shape 항상 동일이라 cache key 1개. MPS는 chi truncation 결과
        shape 변동 시 새 key — 보통 BP iteration 몇 번 후 안정화.
        """
        bix = self.edges[(i, j) if i < j else (j, i)]
        local_tn = self.local_tns[i]

        # sender 부분 — local_tn의 텐서들 (구조 고정, BP 내내 불변)
        sender_inds = [tuple(t.inds) for t in local_tn.tensors]
        sender_shapes = [tuple(t.shape) for t in local_tn.tensors]
        inputs: List[Tuple[str, ...]] = list(sender_inds)
        shapes: List[Tuple[int, ...]] = list(sender_shapes)
        arrays: List[np.ndarray] = [np.asarray(t.data) for t in local_tn.tensors]

        # incoming — canonical 이름으로 매핑하여 cache 친화적
        inc_meta: List[Tuple] = []
        inc_idx = 0
        for k in self.neighbors[i]:
            if k == j:
                continue
            ki_bix = self.edges[(k, i) if k < i else (i, k)]
            c_inds, c_shapes, c_arrays = self._msg_as_canonical_inputs(
                self.messages[(k, i)], ki_bix, inc_idx,
            )
            inputs.extend(c_inds)
            shapes.extend(c_shapes)
            arrays.extend(c_arrays)
            inc_meta.append((tuple(ki_bix), tuple(c_inds), tuple(c_shapes)))
            inc_idx += 1

        cache_key = (
            i, j, tuple(bix),
            tuple(sender_inds), tuple(sender_shapes),
            tuple(inc_meta),
        )
        expr = self._expr_cache.get(cache_key)
        if expr is None:
            expr = ctg.array_contract_expression(
                inputs=inputs,
                output=tuple(bix),
                shapes=shapes,
                optimize="greedy",
            )
            self._expr_cache[cache_key] = expr

        arr = np.ascontiguousarray(expr(*arrays))

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
    expr_cache: Optional[Dict[Tuple, object]] = None,
    **contract_opts,
):
    """BlockBPMps 만들고 run + contract 한 번에 — vanilla `contract_l1bp` 시그니처와 호환.

    `expr_cache`: 외부 dict 주면 cotengra path 표현 재사용 (cross-shot caching). 같은
    (d, k, partition, max_chi)에서 syndrome만 바뀌는 시나리오에 큰 이득.
    """
    bp = BlockBPMps(
        tn,
        site_tags=site_tags,
        max_chi=max_chi,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        optimize=optimize,
        expr_cache=expr_cache,
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
