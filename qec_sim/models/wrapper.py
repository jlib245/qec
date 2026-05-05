# qec_sim/models/wrapper.py
import torch
import torch.nn as nn
from qec_sim.core.interfaces import BasePreprocessor


class PreprocessorWrapper(nn.Module):
    """
    코어 모델 + 전처리기를 하나의 nn.Module로 묶는 래퍼.

    - Trainer는 이 래퍼에 dict 배치만 던지면 됩니다.
    - preprocessor는 nn.Module이지만 모든 텐서를 register_buffer(persistent=False)로
      등록하므로 state_dict()에는 core_model 가중치만 포함됩니다.
    - .to(device) 호출 시 preprocessor의 인덱스 텐서도 자동 이동.
    """

    def __init__(self, core_model: nn.Module, preprocessor: BasePreprocessor):
        super().__init__()
        self.core_model = core_model
        self.preprocessor = preprocessor

    def forward(self, batch_data: dict) -> torch.Tensor:
        x = self.preprocessor.gpu_transform(batch_data)
        return self.core_model(x)
