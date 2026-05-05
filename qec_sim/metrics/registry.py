import torch.nn as nn

from qec_sim.core.registry import Registry

criterion_registry: Registry[nn.Module] = Registry("criterion")

# PyTorch 내장 클래스라 데코레이터로 등록 못 함 — register()를 직접 호출.
criterion_registry.register("bce_with_logits")(nn.BCEWithLogitsLoss)
criterion_registry.register("cross_entropy")(nn.CrossEntropyLoss)
criterion_registry.register("mse")(nn.MSELoss)


def build_criterion(name: str, **kwargs) -> nn.Module:
    """yaml의 이름으로 Loss 함수 인스턴스를 생성."""
    return criterion_registry.get(name)(**kwargs)
