"""Tesseract search-based decoder (quantumlib/tesseract-decoder).

A* + Dijkstra over the DEM hypergraph. Strong baseline for qLDPC (BB) circuits;
also runs on surface code DEMs. Installed via PyPI `tesseract-decoder`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import stim

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("tesseract")
class TesseractDecoder(BaseDecoder):
    def __init__(
        self,
        error_model: stim.DetectorErrorModel,
        circuit: Optional[stim.Circuit] = None,
        det_beam: int = 5,
        beam_climbing: bool = False,
        no_revisit_dets: bool = True,
        merge_errors: bool = True,
        det_penalty: float = 0.0,
        pqlimit: int = 200_000,
        **kwargs,
    ):
        super().__init__(circuit=circuit)
        from tesseract_decoder import tesseract as _ts

        cfg = _ts.TesseractConfig(
            dem=error_model,
            det_beam=det_beam,
            beam_climbing=beam_climbing,
            no_revisit_dets=no_revisit_dets,
            merge_errors=merge_errors,
            det_penalty=det_penalty,
            pqlimit=pqlimit,
        )
        self._dec = _ts.TesseractDecoder(cfg)
        self._n_obs = self._dec.num_observables

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return self._dec.decode_batch(syndromes.astype(bool))

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        err_ids = self._dec.decode_to_errors(syndrome.astype(bool))
        obs = self._dec.get_observables_from_errors(err_ids)
        return {
            "logical_error": np.asarray(obs, dtype=bool).tolist(),
            "corrected_fault_ids": [int(i) for i in err_ids],
        }
