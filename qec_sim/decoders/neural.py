# qec_sim/decoders/neural.py
import torch
import numpy as np

from .base import BaseDecoder
from .registry import register_decoder
from .lut import compute_lut_correction, coset_index_to_bits
from qec_sim.models.wrapper import PreprocessorWrapper


@register_decoder("neural_decoder")
class NeuralDecoder(BaseDecoder):
    """
    PreprocessorWrapper를 감싸는 디코더.
    numpy 배열을 받아 dict 텐서로 변환 후 래퍼에 전달합니다.

    coset_mode=True일 때:
      CNN → 4-class logits → argmax → coset class → LUT correction과 XOR → 최종 observable 예측
    """

    def __init__(self, model: PreprocessorWrapper, coset_lut=None, **kwargs):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.coset_lut = coset_lut

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None, batch_size: int = 4096) -> np.ndarray:
        n = len(syndromes)
        all_preds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_dict = {
                'syndromes': torch.tensor(syndromes[start:end], dtype=torch.float32).to(self.device)
            }
            if erasures is not None:
                batch_dict['erasures'] = torch.tensor(erasures[start:end], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_dict)

                if self.coset_lut is not None:
                    coset_pred = logits.argmax(dim=-1).cpu().numpy()
                    coset_bits = coset_index_to_bits(coset_pred, self.coset_lut.shape[1])
                    lut_corr = compute_lut_correction(syndromes[start:end], self.coset_lut)
                    all_preds.append((coset_bits ^ lut_corr).astype(bool))
                else:
                    # binary mode
                    all_preds.append((logits > 0).cpu().numpy().astype(bool))

        return np.concatenate(all_preds, axis=0)

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        preds = self.decode_batch(syndrome[np.newaxis], batch_size=1)
        return {
            "logical_error": preds[0].tolist(),
            "corrected_fault_ids": [],
        }
