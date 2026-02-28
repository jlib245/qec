# qec_sim/models/registry.py
import inspect
import torch.nn as nn
from typing import Type

MODEL_REGISTRY = {}

def register_model(name: str):
    def wrapper(cls: Type[nn.Module]):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def build_model(name: str, **kwargs) -> nn.Module:
    """이름표를 기반으로 모델 인스턴스를 생성하여 반환합니다."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}'이(가) 등록되지 않았습니다! 사용 가능한 모델: {available}")
    
    cls = MODEL_REGISTRY[name]
    
    # 클래스의 __init__ 시그니처를 분석하여 요구하는 파라미터만 필터링
    sig = inspect.signature(cls)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # 걸러진 파라미터만 모델에 전달
    return cls(**filtered_kwargs)