import torch
import torch.nn as nn
from .registry import register_model

@register_model("jung_cnn_low")
class JungCNN_Low(nn.Module):
    """
    CNN-based decoder based on Jung et al., IEEE TQE 2024.
    [Stim 적용] H 행렬 대신 stim의 디텍터 시공간 좌표(x, y, t)를 사용하여 그리드를 매핑.
    """
    def __init__(self, detector_coords: dict, num_observables: int, code_distance: int, num_detectors: int):
        """
        Args:
            detector_coords (dict): stim.Circuit().get_detector_coordinates() 의 반환값 
                                    예: {0: [2.0, 1.0, 0.0], 1: [4.0, 1.0, 0.0], ...} (x, y, t)
            num_observables (int): 예측해야 할 논리적 관측값(Logical Observables)의 수
            code_distance (int): L (Surface code distance)
            num_detectors (int): 총 디텍터 수 (입력 신드롬 텐서의 크기)
        """
        super().__init__()
        self.L = code_distance
        self.num_detectors = num_detectors
        
        # 1. Stim 좌표계 분석을 통한 Grid 크기 및 채널 깊이(Time) 자동 계산
        all_x = [coords[0] for coords in detector_coords.values()]
        all_y = [coords[1] for coords in detector_coords.values()]
        all_t = [coords[2] for coords in detector_coords.values()] if len(list(detector_coords.values())[0]) > 2 else [0]
        
        # 좌표를 0부터 시작하는 정수 인덱스로 정규화 (보통 stim 좌표는 float이거나 간격이 존재)
        self.min_x, self.min_y = min(all_x), min(all_y)
        self.x_step = 2.0 # Surface code의 일반적인 공간 간격 (환경에 맞게 조정 가능)
        self.y_step = 2.0
        
        self.grid_w = int((max(all_x) - self.min_x) // self.x_step) + 1
        self.grid_h = int((max(all_y) - self.min_y) // self.y_step) + 1
        
        # 시간축(Time) 스텝 수 계산 = Depth
        unique_t = sorted(list(set(all_t)))
        self.t_map = {t_val: i for i, t_val in enumerate(unique_t)}
        self.input_depth = len(unique_t)
        
        # 2. 매핑 테이블 생성 (디텍터 인덱스 -> (t_idx, row, col))
        self.mapping = []
        for det_idx in range(num_detectors):
            if det_idx in detector_coords:
                x, y = detector_coords[det_idx][0], detector_coords[det_idx][1]
                t = detector_coords[det_idx][2] if len(detector_coords[det_idx]) > 2 else 0
                
                col = int((x - self.min_x) // self.x_step)
                row = int((y - self.min_y) // self.y_step)
                t_idx = self.t_map[t]
                
                self.mapping.append((det_idx, t_idx, row, col))
        
        # in_channels: 각 시간 스텝별로 Z, X (또는 단일 측정) 타입이 섞여 들어오므로
        # 가장 범용적으로는 input_depth * 1 채널을 기본으로 하거나, (기존처럼 Z/X 분리 시 * 2)
        # 여기서는 Stim 데이터 구조상 time-step 별로 1개의 채널로 겹쳐서 쌓음.
        # ->수정: 신드롬(Time) + 이레이저(Time) 이므로 채널 수를 2배로 늘림
        in_channels = self.input_depth * 2

        # CNN 필터 설정 (기존 논문 하이퍼파라미터 유지)
        n_filters = 8 if self.L == 3 else (32 if self.L == 5 else 64)
            
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        
        out_h = self.grid_h + 1
        out_w = self.grid_w + 1
        self.flatten_dim = n_filters * out_h * out_w
        
        self.fc1 = nn.Linear(self.flatten_dim, 50)
        self.fc2 = nn.Linear(50, num_observables)

    def _syndrome_to_grid(self, x):
        """
        x shape: (Batch, 2, num_detectors)
        x[:, 0, :] -> Syndromes (신드롬 정보)
        x[:, 1, :] -> Erasures (누출 에러 정보)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 채널 수가 2배인 빈 캔버스 준비 (배경은 -0.5)
        grid = torch.full((batch_size, self.input_depth * 2, self.grid_h, self.grid_w), -0.5, device=device)
        
        for (det_idx, t_idx, row, col) in self.mapping:
            # 1. 신드롬 값을 해당 (t_idx) 채널에 채우기
            grid[:, t_idx, row, col] = x[:, 0, det_idx]
            
            # 2. 이레이저 값을 뒤쪽 (self.input_depth + t_idx) 채널에 채우기
            grid[:, self.input_depth + t_idx, row, col] = x[:, 1, det_idx]
            
        return grid

    def forward(self, x):
        # x shape expected: (Batch, num_detectors)
        x = self._syndrome_to_grid(x) 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x