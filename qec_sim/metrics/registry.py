# qec_sim/metrics/registry.py
import torch.nn as nn

# Loss 함수를 관리하는 레지스트리
CRITERION_REGISTRY = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss
}

def build_criterion(name: str, **kwargs):
    """YAML의 이름을 바탕으로 Loss 함수 인스턴스를 생성함"""
    if name not in CRITERION_REGISTRY:
        raise ValueError(f"지원하지 않는 Loss: {name}. 가능 목록: {list(CRITERION_REGISTRY.keys())}")
    return CRITERION_REGISTRY[name](**kwargs)