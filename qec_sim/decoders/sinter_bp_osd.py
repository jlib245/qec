"""sinter용 BP+OSD adapter.

oscarhiggott/BeliefMatching의 `BeliefMatchingSinterDecoder` 패턴 재사용 — DEM에서
ldpc.BpOsdDecoder를 직접 빌드하고 b8 file 경로 인터페이스에 맞춰 매 task마다 재구성.
algo decoder라 multiprocess 친화적 (per-shot Python loop이지만 worker 병렬로 충분).
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import numpy as np
import sinter
import stim


class BpOsdSinterDecoder(sinter.Decoder):
    """sinter benchmark 인터페이스로 노출되는 BP+OSD decoder."""

    def __init__(
        self,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "OSD_0",
        osd_order: int = 0,
        **kwargs: Any,
    ) -> None:
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order
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
        from ldpc.bposd_decoder import BpOsdDecoder as _LdpcBpOsd

        dem = stim.DetectorErrorModel.from_file(dem_path)
        mats = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        dec = _LdpcBpOsd(
            mats.check_matrix,
            error_channel=list(mats.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
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
