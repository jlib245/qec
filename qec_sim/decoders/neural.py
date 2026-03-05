# qec_sim/decoders/neural.py
import torch
import numpy as np
from .base import BaseDecoder
from .registry import register_decoder

@register_decoder("neural_decoder")
class NeuralDecoder(BaseDecoder):
    def __init__(self, model: torch.nn.Module, **kwargs): 
        """
        외부에서 이미 생성되고 가중치가 로드된 모델 객체를 주입받습니다.
        MWPM용 error_model 등 안 쓰는 인자는 **kwargs가 흡수합니다.
        """
        self.model = model
        self.model.eval()
        
        # 주입받은 모델의 파라미터로부터 현재 디바이스(CPU/GPU)를 자동으로 추론
        self.device = next(self.model.parameters()).device

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        if erasures is None:
            erasures = np.zeros_like(syndromes)
            
        # Numpy 배열을 (Batch, 2채널, Detectors) 형태의 파이토치 텐서로 변환
        x = np.stack([syndromes, erasures], axis=1)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            # Logit이 0보다 크면 에러(True), 아니면 정상(False)으로 예측
            predictions = (logits > 0).cpu().numpy().astype(bool)
            
        return predictions