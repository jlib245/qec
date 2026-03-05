# qec_sim/circuit/registry.py

_BUILDER_REGISTRY = {}

def register_builder(name):
    """빌더 클래스를 레지스트리에 등록하는 데코레이터"""
    def decorator(cls):
        _BUILDER_REGISTRY[name] = cls
        return cls
    return decorator

def build_circuit(name, code_config, noise_config, **kwargs):
    """이름으로 등록된 빌더를 찾아 인스턴스를 생성하는 함수"""
    if name not in _BUILDER_REGISTRY:
        raise ValueError(f"'{name}' 이름의 회로 빌더가 등록되지 않았습니다. "
                         f"사용 가능한 빌더: {list(_BUILDER_REGISTRY.keys())}")
    
    builder_class = _BUILDER_REGISTRY[name]
    return builder_class(code_config=code_config, noise_config=noise_config, **kwargs)