"""BP+OSD decoder via Roffe's `ldpc` package.

Surface code뿐 아니라 일반 qLDPC code 평가에도 활용 가능. DEM의 hyperedge를 그대로
사용 (decompose_errors=False) — BP는 hyperedge graph에서 자연스러움. observable
prediction은 매칭 결과 error vector에 observables_matrix를 곱해서 도출.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import stim

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("bp_osd")
class BpOsdDecoder(BaseDecoder):
    """Roffe's `ldpc.BpOsdDecoder`를 BaseDecoder로 래핑.

    `error_model`은 가능하면 `decompose_errors=False`로 만든 DEM 권장 — BP+OSD는
    hyperedge에 잘 작동하므로 분해된 그래프보다 정보 손실이 적음. 다만 본 클래스는
    어느 쪽이든 받음 (`allow_undecomposed_hyperedges=True`).
    """

    def __init__(
        self,
        error_model: stim.DetectorErrorModel,
        circuit: Optional[stim.Circuit] = None,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "OSD_0",
        osd_order: int = 0,
        **kwargs,
    ):
        super().__init__(circuit=circuit)
        from beliefmatching import detector_error_model_to_check_matrices
        from ldpc.bposd_decoder import BpOsdDecoder as _LdpcBpOsd

        self.matrices = detector_error_model_to_check_matrices(
            error_model, allow_undecomposed_hyperedges=True
        )
        self._dec = _LdpcBpOsd(
            self.matrices.check_matrix,
            error_channel=list(self.matrices.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            osd_method=osd_method,
            osd_order=osd_order,
        )

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        n_obs = self.matrices.observables_matrix.shape[0]
        preds = np.zeros((syndromes.shape[0], n_obs), dtype=bool)
        for i in range(syndromes.shape[0]):
            err = self._dec.decode(syndromes[i].astype(np.uint8))
            preds[i] = ((self.matrices.observables_matrix @ err) % 2).astype(bool)
        return preds

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        err = self._dec.decode(syndrome.astype(np.uint8))
        obs = ((self.matrices.observables_matrix @ err) % 2).astype(bool)
        # error vector의 nonzero 위치를 fault id로 노출 (시각화는 부분적)
        fault_ids = [int(i) for i, v in enumerate(err) if v]
        return {
            "logical_error": obs.tolist(),
            "corrected_fault_ids": fault_ids,
        }
