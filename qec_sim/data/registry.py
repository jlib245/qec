# qec_sim/data/registry.py 
from typing import Type

PREPROCESSOR_REGISTRY = {}

def register_preprocessor(name: str):
    def wrapper(cls: Type):
        PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return wrapper

def build_preprocessor(name: str, **kwargs):
    if name not in PREPROCESSOR_REGISTRY:
        available = list(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(f"Preprocessor '{name}'이(가) 등록되지 않았습니다! 사용 가능: {available}")
    
    # 기본값이 없으므로 kwargs에서 필수 인자가 누락되면 파이썬 단에서 에러를 뱉어줍니다.
    return PREPROCESSOR_REGISTRY[name](**kwargs)