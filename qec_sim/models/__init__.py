# qec_sim/decoders/__init__.py 

# 레지스트리 관련 함수 노출
from .registry import build_model, register_model

# 시스템에 등록할 신경망 모델들을 여기서 임포트해 줍니다.
from .baseline import ErasureAwareMLP
# 나중에 CNN, Transformer 등을 만들면 여기에 추가