# qec_sim/metrics/evaluator.py
import torch
import torch.nn as nn


def coerce_label_dtype(y: torch.Tensor) -> torch.Tensor:
    """CrossEntropyLoss는 long, BCEWithLogitsLoss는 float을 요구."""
    return y.long() if y.dtype in (torch.int32, torch.int64) else y.float()


class Evaluator:
    def __init__(self, device, criterion=None):
        self.device = device
        self.criterion = criterion

    def validate_on_loader(self, model: nn.Module, loader, val_steps) -> tuple:
        """[Trainer 용] PreprocessorWrapper 모델과 DataLoader를 이용한 검증.
        DataLoader는 (dict_batch, labels) 형태의 배치를 반환해야 합니다.
        """
        if self.criterion is None:
            raise ValueError("validate_on_loader를 사용하려면 criterion이 주입되어야 합니다.")

        model.eval()
        total_loss = 0.0
        total_errors = 0
        total_samples = 0
        num_steps = 0

        with torch.no_grad():
            for step, (batch_dict, labels) in enumerate(loader):
                if val_steps is not None and step >= val_steps:
                    break

                batch_data = {k: v.to(self.device).float() for k, v in batch_dict.items()}
                batch_y = coerce_label_dtype(labels.to(self.device))

                outputs = model(batch_data)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                if batch_y.dtype == torch.long:
                    # coset mode: 4-class argmax
                    preds = outputs.argmax(dim=-1)
                    batch_errors = (preds != batch_y).sum().item()
                else:
                    # binary mode
                    preds = (outputs > 0.0).float()
                    batch_errors = (preds.bool() != batch_y.bool()).any(dim=1).sum().item()
                total_errors += batch_errors
                total_samples += batch_y.size(0)
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        ler = total_errors / max(total_samples, 1)
        return avg_loss, ler
