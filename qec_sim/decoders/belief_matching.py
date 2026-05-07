# qec_sim/decoders/belief_matching.py
"""BaseDecoder wrapper around `oscarhiggott/BeliefMatching` (Higgott et al. PRX 2023).

Pipeline: stim DEM (decompose_errors=True) → ldpc.bp_decoder posteriors → PyMatching
edge weights re-derived from posteriors → matching. 외부 패키지 `beliefmatching`이
모든 무거운 일을 수행하므로 본 클래스는 얇은 어댑터.
"""

from typing import Optional

import numpy as np
import stim

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("belief_matching")
class BeliefMatchingDecoder(BaseDecoder):
    """oscarhiggott/BeliefMatching를 BaseDecoder 인터페이스에 맞춰 래핑.

    DEM은 반드시 `decompose_errors=True`로 만들어진 것이어야 함 — BM이 hyperedge를
    edge로 분해된 형태를 가정.
    """

    def __init__(
        self,
        error_model: stim.DetectorErrorModel,
        circuit: Optional[stim.Circuit] = None,
        max_bp_iters: int = 20,
        bp_method: str = "product_sum",
        **kwargs,
    ):
        super().__init__(circuit=circuit)
        # 지연 import — 의존성 없는 환경에서 모듈 로드만으로 깨지지 않도록
        from beliefmatching import BeliefMatching

        self.error_model = error_model
        self._bm = BeliefMatching(
            error_model, max_bp_iters=max_bp_iters, bp_method=bp_method
        )

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return self._bm.decode_batch(syndromes.astype(np.uint8))

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        # BM 패키지는 per-mechanism fault_id를 노출하지 않으므로 시각화는 미지원.
        logical_error = self._bm.decode(syndrome.astype(np.uint8))
        return {
            "logical_error": logical_error.tolist(),
            "corrected_fault_ids": [],
        }
