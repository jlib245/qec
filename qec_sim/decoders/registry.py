# qec_sim/decoders/registry.py
from typing import Type
from .base import BaseDecoder

DECODER_REGISTRY = {}

def register_decoder(name: str):
    """디코더를 레지스트리에 등록하는 데코레이터"""
    def wrapper(cls: Type[BaseDecoder]):
        DECODER_REGISTRY[name] = cls
        return cls
    return wrapper

def build_decoder(name: str, **kwargs) -> BaseDecoder:
    """이름표를 기반으로 디코더 인스턴스를 생성하여 반환합니다."""
    if name not in DECODER_REGISTRY:
        available = list(DECODER_REGISTRY.keys())
        raise ValueError(f"Decoder '{name}'이(가) 등록되지 않았습니다! 사용 가능한 디코더: {available}")
    return DECODER_REGISTRY[name](**kwargs)