import torch
import numpy as np
from .base import BaseDecoder
from .registry import register_decoder

# 모델 레지스트리에서 모델을 동적으로 불러오기 위해 임포트
from qec_sim.models.registry import build_model

@register_decoder("neural_decoder")
class NeuralDecoder(BaseDecoder):
    def __init__(self, 
                 model_name: str, 
                 num_detectors: int, 
                 num_observables: int, 
                 model_kwargs: dict = None, 
                 weight_path: str = None, 
                 **kwargs): # MWPM용 error_model 등 안 쓰는 인자를 무시하기 위한 **kwargs
        
        if model_kwargs is None:
            model_kwargs = {}
            
        # 1. 레지스트리에서 설정된 이름의 모델 생성 (예: "erasure_mlp")
        self.model = build_model(model_name, num_detectors=num_detectors, num_observables=num_observables, **model_kwargs)
        
        # 2. 저장된 가중치(학습된 모델)가 있다면 불러오기
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))
            
        # 3. 평가 모드 및 디바이스 설정
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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