"""sinter용 BP+LSD adapter.

ldpc.BpLsdDecoder를 직접 감싼다. ldpc 기본 제공 SinterLsdDecoder는 `lsd_method`를
노출하지 않아 항상 LSD_0(order 0)로만 동작 — order>0(LSD_CS/LSD_E)을 쓰려면
`lsd_method`를 넘겨야 하므로 자체 adapter가 필요하다. sinter_bp_osd.py와 동일 구조.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import numpy as np
import sinter
import stim


class BpLsdSinterDecoder(sinter.Decoder):
    """sinter benchmark 인터페이스로 노출되는 BP+LSD decoder.

    lsd_method: 'LSD_0'(order 강제 0) / 'LSD_CS'(combination sweep) / 'LSD_E'(exhaustive).
                order>0을 쓰려면 CS 또는 E여야 한다 (OSD의 osd_method와 동일 규칙).
    """

    def __init__(
        self,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        lsd_method: str = "LSD_0",
        lsd_order: int = 0,
        **kwargs: Any,
    ) -> None:
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.lsd_method = lsd_method
        self.lsd_order = lsd_order
        self.kwargs: Dict[str, Any] = kwargs

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        from beliefmatching import detector_error_model_to_check_matrices
        from ldpc.bplsd_decoder import BpLsdDecoder

        dem = stim.DetectorErrorModel.from_file(dem_path)
        mats = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        dec = BpLsdDecoder(
            mats.check_matrix,
            error_channel=list(mats.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            lsd_method=self.lsd_method,
            lsd_order=self.lsd_order,
            **self.kwargs,
        )

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        )

        n_obs_out = mats.observables_matrix.shape[0]
        predictions = np.zeros((num_shots, n_obs_out), dtype=bool)
        for i in range(num_shots):
            err = dec.decode(shots[i].astype(np.uint8))
            predictions[i] = ((mats.observables_matrix @ err) % 2).astype(bool)

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )
