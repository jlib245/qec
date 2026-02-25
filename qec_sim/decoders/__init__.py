# qec_sim/decoders/__init__.py 

# 레지스트리 관련 함수 노출
from .registry import build_decoder, register_decoder

# 시스템에 등록할 디코더들을 여기서 미리 한 번씩 임포트해 줍니다.
from .mwpm import ErasureMWPM
# 나중에 새로운 디코더(예: bp.py)를 만들면 여기에 from .bp import ... 추가
from .neural import NeuralDecoder