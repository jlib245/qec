# Quantum Error Correction Simulator

## 실행 예시

### NN디코더 학습
```sh
uv run python tests/baseline/train.py --config configs/experiment_mlp.yaml
```

### 디코더 성능 평가
```sh
uv run python tests/baseline/run_experiment.py --config configs/experiment_mwpm.yaml
```

## 확장 가이드
### NN 디코더 추가
1. `qec_sim/models/` 폴더에 파이썬 파일(예: `cnn.py`)을 생성하고 모델을 구현합니다.
2. 클래스 선언부 위에 `@register_model("이름")` 데코레이터를 붙입니다.
```python
from .registry import register_model
import torch.nn as nn

@register_model("erasure_cnn")
class ErasureCNN(nn.Module):
    def __init__(self, num_detectors, num_observables, **kwargs):
        super().__init__()
        # ... 모델 구조 구현 ...
```
3. `qec_sim/models/__init__.py` 파일에 `from .cnn import ErasureCNN`을 추가하여 시스템에 노출시킵니다.
3. YAML 파일에서 `model: name: "erasure_cnn"`으로 변경 후 스크립트를 실행합니다.

### 고전 디코더 추가
1. `qec_sim/decoders/` 폴더에 파일(예: `bp.py`)을 생성하고 `BaseDecoder`를 상속받습니다.
2. 클래스 선언부 위에 `@register_decoder("이름")` 데코레이터를 붙입니다.
```python from .base import BaseDecoder
from .registry import register_decoder

@register_decoder("bp_decoder")
class BeliefPropagationDecoder(BaseDecoder):
    def __init__(self, error_model, **kwargs):
        self.error_model = error_model

    def decode_batch(self, syndromes, erasures=None):
        # ... 디코딩 알고리즘 구현 ...
        return predictions
```
3. `qec_sim/decoders/__init__.py` 파일에 추가하고, YAML 파일에서 `decoder: name: "bp_decoder"`로 변경하여 즉시 테스트합니다.

## 설정파일(YAML) 구조
모든 실험 환경은 configs/ 폴더의 YAML 파일에서 중앙 통제됩니다. 파이썬 코드 수정 없이 설정 파일만으로 벤치마크를 수행할 수 있습니다.
```YAML
# configs/experiment_mlp.yaml
simulation:
  shots: 1000

code:
  distance: 5
  rounds: 5

noise:
  p_gate: 0.005
  p_meas: 0.005
  p_corr: 0.001
  p_leak: 0.03

training:
  data_mode: "offline"       # 실시간 데이터 생성 시 "online" 사용
  train_path: "datasets/d5_train.npz"
  val_path: "datasets/d5_val.npz"
  batch_size: 512
  epochs: 20
  learning_rate: 0.001
  save_path: "checkpoints/weights.pth"

model:
  name: "erasure_mlp"        # 레지스트리에 등록된 딥러닝 모델 이름
  kwargs:
    hidden_dim: 512

decoder:
  name: "neural_decoder"     # 딥러닝 모델용 래퍼 디코더
  model_name: "erasure_mlp"  # 내부에 장착할 실제 모델
  model_kwargs:
    hidden_dim: 512
  weight_path: "checkpoints/weights.pth" # 평가 시 불러올 학습된 가중치 경로
```