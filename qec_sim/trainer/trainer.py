import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, evaluator, train_loader, val_loader, optimizer, scheduler, callbacks, train_steps, val_steps):
        """
        오직 모델의 학습(Train)과 상태 관리만 담당하는 트레이너 클래스.
        평가는 Evaluator에, 파일 저장 및 로깅은 Callbacks에 위임합니다.
        """
        self.model = model
        self.evaluator = evaluator
        self.device = evaluator.device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.callbacks = callbacks
        self.stop_training = False
        self.train_steps = train_steps
        self.val_steps = val_steps

    def train_epoch(self):
        """한 에포크 동안의 순전파 및 역전파(Backpropagation) 수행"""
        self.model.train()
        total_loss = 0.0
        
        for step, (batch_x, batch_y) in enumerate(self.train_loader):
            # 지정된 스텝 수 에 도달하면 조기 종료 (옵션)
            if self.train_steps is not None and step >= self.train_steps:
                break

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.evaluator.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def fit(self, epochs):
        """전체 학습 루프 실행 및 기존 구동 방식 완벽 호환"""
        # [Hook] 학습 시작 전 콜백 실행
        for cb in self.callbacks: 
            cb.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training: 
                break
                
            # [Hook] 에포크 시작 전 콜백 실행
            for cb in self.callbacks: 
                cb.on_epoch_begin(self, epoch)

            # 1. 학습 및 검증 수행
            train_loss = self.train_epoch()
            val_loss, val_ler = self.evaluator.validate_on_loader(
                self.model, 
                self.val_loader, 
                self.val_steps
            )
            # 2. 현재 학습률(LR) 가져오기
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 3. 콜백에 넘겨줄 로그 데이터 정리
            logs = {
                'lr': current_lr, 
                'train_loss': train_loss, 
                'val_loss': val_loss, 
                'val_ler': val_ler
            }
            
            # 4. 기존 깃허브와 동일한 직관적인 콘솔 출력
            print(f"[Epoch {epoch+1:02d}/{epochs}] LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LER: {val_ler * 100:.2f}%")

            # 5. 스케줄러 업데이트 (기존 로직 보존)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # [Hook] 에포크 종료 후 콜백 실행 (CSV 기록, 모델 저장, 조기 종료 체크)
            for cb in self.callbacks: 
                cb.on_epoch_end(self, epoch, logs)

        # [Hook] 학습 완전히 종료 후 콜백 실행
        for cb in self.callbacks: 
            cb.on_train_end(self)