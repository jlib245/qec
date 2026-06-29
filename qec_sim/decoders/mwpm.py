# qec_sim/decoders/mwpm.py
import pymatching
import stim
import numpy as np

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("mwpm")
class MWPMDecoder(BaseDecoder):
    def __init__(self, error_model: stim.DetectorErrorModel, circuit: stim.Circuit = None, **kwargs):
        super().__init__(circuit=circuit)
        self.error_model = error_model
        self.base_matching = pymatching.Matching.from_detector_error_model(error_model)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return self.base_matching.decode_batch(syndromes)

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        logical_error = self.base_matching.decode(syndrome)
        edges = self.base_matching.decode_to_edges_array(syndrome)
        corrected_fault_ids = list({int(fid) for edge in edges for fid in edge if fid != -1})
        return {
            "logical_error": logical_error.tolist(),
            "corrected_fault_ids": corrected_fault_ids,
        }
