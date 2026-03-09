# qec_sim/models/jung_cnn_low.py
import torch
import torch.nn as nn
from .registry import register_model

@register_model("jung_cnn_low")
class JungCNN_Low(nn.Module):
    """
    CNN-based decoder based on Jung et al., IEEE TQE 2024.
    (데이터 전처리 및 그리드 변환 로직은 GridPreprocessor로 위임되었습니다.)
    """
    def __init__(self, in_channels: int, grid_h: int, grid_w: int, num_observables: int, n_filters: int):
        """
        Args:
            in_channels (int): 입력 텐서의 채널 수 (Preprocessor가 결정)
            grid_h (int): 입력 텐서의 높이 (Preprocessor가 결정)
            grid_w (int): 입력 텐서의 너비 (Preprocessor가 결정)
            num_observables (int): 최종 출력 클래스 수
            n_filters (int): CNN 필터 수 (YAML에서 반드시 명시해야 함)
        """
        super().__init__()
        
        # 2. 합성곱 신경망(CNN) 계층 정의
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        
        # 풀링(Pooling)이나 스트라이드(Stride)를 쓰지 않았으므로 공간 크기(H, W)는 유지됨
        self.flatten_dim = n_filters * grid_h * grid_w
        
        # 3. 완전 연결(FC) 계층 정의
        self.fc1 = nn.Linear(self.flatten_dim, 50)
        self.fc2 = nn.Linear(50, num_observables)

    def forward(self, x):
        """
        x shape expected: (Batch, in_channels, grid_h, grid_w)
        기존의 _syndrome_to_grid 호출은 삭제되었으며, 이미 그리드 형태로 들어옵니다.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # (Batch, Flatten) 형태로 펼치기
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x