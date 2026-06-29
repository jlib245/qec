# qec_sim/trainer/trainer.py
import os
import torch
from qec_sim.metrics.evaluator import coerce_label_dtype


# Env-driven knobs (default off to avoid breaking previous runs):
#   QEC_USE_COMPILE=1     → torch.compile core model (1.5–2× via kernel fusion)
#   QEC_AMP_DTYPE=bf16    → bfloat16 mixed-precision autocast (RTX 4090 native)
_USE_COMPILE = os.environ.get("QEC_USE_COMPILE", "0") == "1"
_AMP_DTYPE_STR = os.environ.get("QEC_AMP_DTYPE", "").lower()
_AMP_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}.get(_AMP_DTYPE_STR, None)


def _autocast_ctx(device: torch.device):
    """Return autocast context if AMP enabled and on CUDA, else null context."""
    if _AMP_DTYPE is None or device.type != "cuda":
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=_AMP_DTYPE)


class Trainer:
    def __init__(self, wrapped_model, evaluator, train_loader, val_loader,
                 optimizer, scheduler, callbacks, train_steps, val_steps):
        # Opt-in compile of the heavy compute (core model only — preprocessor's
        # scatter ops sometimes confuse torch.compile dynamic shape detection).
        if _USE_COMPILE and hasattr(wrapped_model, "core_model"):
            wrapped_model.core_model = torch.compile(
                wrapped_model.core_model, mode="reduce-overhead"
            )
            print(f"  [trainer] torch.compile enabled on core_model")
        if _AMP_DTYPE is not None:
            print(f"  [trainer] AMP enabled with dtype={_AMP_DTYPE}")

        self.model = wrapped_model
        self.evaluator = evaluator
        self.device = evaluator.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.stop_training = False

    def train_epoch(self):
        from tqdm.auto import tqdm
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        # train_steps가 있으면 그걸 total로, 아니면 loader 길이 (IterableDataset은 len 없음).
        if self.train_steps:
            total = self.train_steps
        else:
            try:
                total = len(self.train_loader)
            except TypeError:
                total = None
        pbar = tqdm(self.train_loader, total=total, desc='train', leave=False, dynamic_ncols=True)

        for step, (batch_dict, labels) in enumerate(pbar):
            if self.train_steps and step >= self.train_steps:
                break

            batch_data = {k: v.to(self.device).float() for k, v in batch_dict.items()}
            y = coerce_label_dtype(labels.to(self.device))

            self.optimizer.zero_grad()
            with _autocast_ctx(self.device):
                outputs = self.model(batch_data)
                loss = self.evaluator.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_steps += 1
            if num_steps % 10 == 0:
                pbar.set_postfix(loss=f"{total_loss/num_steps:.4f}")

        return total_loss / max(num_steps, 1)

    def fit(self, epochs: int):
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training:
                break

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            train_loss = self.train_epoch()
            # PreprocessorWrapper가 전처리를 담당하므로 model만 전달
            val_loss, val_ler = self.evaluator.validate_on_loader(
                self.model, self.val_loader, self.val_steps
            )

            current_lr = self.optimizer.param_groups[0]['lr']
            logs = {
                'lr': current_lr,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ler': val_ler,
            }

            print(f"[Epoch {epoch+1:02d}/{epochs}] LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LER: {val_ler * 100:.2f}%")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

        for cb in self.callbacks:
            cb.on_train_end(self)
