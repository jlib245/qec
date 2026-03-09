# qec_sim/decoders/neural.py
import torch
import numpy as np
from .base import BaseDecoder
from .registry import register_decoder
from qec_sim.data.preprocessors import BasePreprocessor

@register_decoder("neural_decoder")
class NeuralDecoder(BaseDecoder):
    def __init__(self, model: torch.nn.Module, preprocessor: BasePreprocessor, **kwargs): 
        """
        외부에서 완전히 조립된 모델과 전처리기를 주입받습니다.
        기본값(None) 없이 명시적으로 주입되어야 합니다.
        """
        self.model = model
        self.preprocessor = preprocessor
        
        self.model.eval()
        
        # 모델 파라미터로부터 디바이스 추론
        self.device = next(self.model.parameters()).device

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        # 전처리기에게 변환 위임.
        # erasures가 None으로 들어와도, preprocessor가 자신의 YAML 설정(use_erasures)에 따라 알아서 처리합니다.
        x_tensor = self.preprocessor.process(syndromes, erasures).to(self.device)
        
        # 2. 모델 추론
        with torch.no_grad():
            logits = self.model(x_tensor)
            predictions = (logits > 0).cpu().numpy().astype(bool)
            
        return predictions