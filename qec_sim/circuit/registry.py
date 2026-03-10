# qec_sim/circuit/registry.py
from qec_sim.core.registry import Registry

builder_registry: Registry = Registry("circuit_builder")

register_builder = builder_registry.register


def build_circuit(name: str, code_config, noise_config, **kwargs):
    """이름으로 빌더를 찾아 인스턴스를 생성합니다."""
    return builder_registry.get(name)(code_params=code_config, noise_params=noise_config, **kwargs)
