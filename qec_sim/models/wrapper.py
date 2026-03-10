import torch
import torch.nn as nn

class PreprocessorWrapper(nn.Module):
    """
    팩토리가 만들어내는 최종 산출물.
    내부에는 '순수 모델'을 품고, 앞단에는 'GPU 전처리 정책'을 두르고 있습니다.
    """
    def __init__(self, core_model: nn.Module, gpu_transform_fn):
        super().__init__()
        self.core_model = core_model
        self.gpu_transform = gpu_transform_fn

    def forward(self, batch_data: dict) -> torch.Tensor:
        # 1. 팩토리가 쥐어준 정책대로 GPU 전처리 실행 (예: 1D dict -> 2D Tensor)
        processed_x = self.gpu_transform(batch_data)
        
        # 2. 내부에 품은 코어 모델 실행
        return self.core_model(processed_x)