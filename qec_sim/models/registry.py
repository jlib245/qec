from typing import Type, Dict
from qec_sim.core.interfaces import BaseQECModel

_MODELS: Dict[str, Type[BaseQECModel]] = {}

def register_model(name: str):
    """모델 클래스 위에 @register_model("이름") 형태로 붙이는 데코레이터"""
    def wrapper(cls: Type[BaseQECModel]):
        _MODELS[name] = cls
        return cls
    return wrapper

def get_model_class(name: str) -> Type[BaseQECModel]:
    """이름으로 등록된 모델 '클래스' 자체를 반환합니다."""
    if name not in _MODELS:
        raise KeyError(f"모델 '{name}'을 찾을 수 없습니다. 등록된 모델: {list(_MODELS.keys())}")
    return _MODELS[name]