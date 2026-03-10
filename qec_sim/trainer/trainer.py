import torch

class Trainer:
    def __init__(self, wrapped_model, evaluator, train_loader, val_loader, optimizer, scheduler, callbacks, train_steps, val_steps):
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
        
        for step, (batch_dict, labels) in enumerate(self.train_loader):
            if self.train_steps and step >= self.train_steps:
                break

            # 딕셔너리 내부의 모든 텐서를 디바이스로 이동
            batch_data = {k: v.to(self.device).float() for k, v in batch_dict.items()}
            y = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # 래퍼를 호출하면 알아서 GPU 전처리 -> 모델 연산이 수행됨
            outputs = self.model(batch_data)
            
            loss = self.evaluator.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def fit(self, epochs: int):
        # 학습 시작 전 콜백
        for cb in self.callbacks: 
            cb.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training: 
                break
                
            # 에포크 시작 전 콜백
            for cb in self.callbacks: 
                cb.on_epoch_begin(self, epoch)

            train_loss = self.train_epoch()
            val_loss, val_ler = self.evaluator.validate_on_loader(self.model, self.preprocessor, self.val_loader, self.val_steps)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logs = {
                'lr': current_lr, 
                'train_loss': train_loss, 
                'val_loss': val_loss, 
                'val_ler': val_ler
            }
            
            print(f"[Epoch {epoch+1:02d}/{epochs}] LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LER: {val_ler * 100:.2f}%")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 에포크 종료 후 콜백 (여기서 CSV 기록, 모델 저장, 조기 종료가 수행됨)
            for cb in self.callbacks: 
                cb.on_epoch_end(self, epoch, logs)

        # 학습 완전 종료 후 콜백
        for cb in self.callbacks: 
            cb.on_train_end(self)