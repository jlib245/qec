# qec_sim/trainer/trainer.py
import torch


class Trainer:
    def __init__(self, wrapped_model, evaluator, train_loader, val_loader,
                 optimizer, scheduler, callbacks, train_steps, val_steps):
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
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        for step, (batch_dict, labels) in enumerate(self.train_loader):
            if self.train_steps and step >= self.train_steps:
                break

            batch_data = {k: v.to(self.device).float() for k, v in batch_dict.items()}
            y = labels.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.evaluator.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_steps += 1

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
