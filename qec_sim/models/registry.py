# qec_sim/models/registry.py
import torch.nn as nn
from typing import Type

MODEL_REGISTRY = {}

def register_model(name: str):
    """모델을 레지스트리에 등록하는 데코레이터"""
    def wrapper(cls: Type[nn.Module]):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def build_model(name: str, **kwargs) -> nn.Module:
    """이름표를 기반으로 모델 인스턴스를 생성하여 반환합니다."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}'이(가) 등록되지 않았습니다! 사용 가능한 모델: {available}")
    return MODEL_REGISTRY[name](**kwargs)