"""sinter용 GPU-accelerated BeliefMatching adapter."""

from __future__ import annotations

import pathlib
from typing import Any

import sinter
import stim


class BeliefMatchingFastSinterDecoder(sinter.Decoder):
    """sinter Decoder 인터페이스로 GPU-accel BeliefMatchingFast 노출.

    Note: sinter는 멀티워커 (multiprocessing)로 호출하는데 각 worker가 GPU에 모델
    올리면 GPU 메모리 경쟁 발생. workers=1 (단일 worker, 큰 batch)이 보통 최적.
    """

    def __init__(
        self,
        max_bp_iters: int = 20,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.max_bp_iters = max_bp_iters
        self.device = device
        self.kwargs = kwargs

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
        from qec_sim.decoders.belief_matching_fast import BeliefMatchingFastDecoder

        dem = stim.DetectorErrorModel.from_file(dem_path)
        dec = BeliefMatchingFastDecoder(
            error_model=dem,
            max_bp_iters=self.max_bp_iters,
            device=self.device,
            **self.kwargs,
        )

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        )

        predictions = dec.decode_batch(shots)

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )
