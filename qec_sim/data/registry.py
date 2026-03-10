from typing import Type, Dict
from qec_sim.core.interfaces import BasePreprocessor

_PREPROCESSORS: Dict[str, Type[BasePreprocessor]] = {}

def register_preprocessor(name: str):
    """전처리기 클래스 위에 @register_preprocessor("이름") 형태로 붙이는 데코레이터"""
    def wrapper(cls: Type[BasePreprocessor]):
        _PREPROCESSORS[name] = cls
        return cls
    return wrapper

def get_preprocessor_class(name: str) -> Type[BasePreprocessor]:
    """이름으로 등록된 전처리기 '클래스' 자체를 반환합니다."""
    if name not in _PREPROCESSORS:
        raise KeyError(f"전처리기 '{name}'을 찾을 수 없습니다. 등록된 전처리기: {list(_PREPROCESSORS.keys())}")
    return _PREPROCESSORS[name]