import torch
import torch.nn as nn

from qec_sim.core.registry import Registry

criterion_registry: Registry[nn.Module] = Registry("criterion")

# PyTorch 내장 클래스라 데코레이터로 등록 못 함 — register()를 직접 호출.
criterion_registry.register("bce_with_logits")(nn.BCEWithLogitsLoss)
criterion_registry.register("cross_entropy")(nn.CrossEntropyLoss)
criterion_registry.register("mse")(nn.MSELoss)


@criterion_registry.register("cbam_aux_ce")
class CBAMAuxCE(nn.Module):
    """CrossEntropy that tolerates (main, aux) tuple outputs from CBAMDecoder.
    When `aux_weight > 0` and the model returns a tuple, an extra CE term
    against a fixed-zero target is added (mirrors the regularization in the
    original training script).

    Also squeezes (B, 1) labels and casts to long so the trainer doesn't need
    to know whether coset_mode coerced the labels already.
    """

    def __init__(self, aux_weight: float = 0.0):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main, aux = outputs
        else:
            main, aux = outputs, None

        if target.dim() > 1:
            target = target.view(-1)
        if target.dtype != torch.long:
            target = target.long()

        loss = self.ce(main, target)
        if aux is not None and self.aux_weight > 0.0:
            zeros = torch.zeros_like(target)
            loss = loss + self.aux_weight * self.ce(aux, zeros)
        return loss


def build_criterion(name: str, **kwargs) -> nn.Module:
    """yaml의 이름으로 Loss 함수 인스턴스를 생성."""
    return criterion_registry.get(name)(**kwargs)
