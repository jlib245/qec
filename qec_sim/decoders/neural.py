# qec_sim/decoders/neural.py
import torch
import numpy as np

from .base import BaseDecoder
from .registry import register_decoder
from qec_sim.models.wrapper import PreprocessorWrapper


@register_decoder("neural_decoder")
class NeuralDecoder(BaseDecoder):
    """
    PreprocessorWrapper를 감싸는 디코더.
    numpy 배열을 받아 dict 텐서로 변환 후 래퍼에 전달합니다.
    """

    def __init__(self, model: PreprocessorWrapper, **kwargs):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        batch_dict = {
            'syndromes': torch.tensor(syndromes, dtype=torch.float32).to(self.device)
        }
        if erasures is not None:
            batch_dict['erasures'] = torch.tensor(erasures, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_dict)
            return (logits > 0).cpu().numpy().astype(bool)
