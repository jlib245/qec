

# qec_sim/core/engine.py
import torch
import torch.nn as nn

class QECEngine:
    def __init__(self, model, criterion, optimizer=None, device='cuda'):
        """
        공통 연산 스텝: 데이터 전처리 -> 추론 -> (학습) -> 지표 계산
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # BCEWithLogitsLoss를 쓰면 임계값 0.0, 아니면 0.5
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            self.threshold = 0.0
        else:
            self.threshold = 0.5

    def step(self, batch_x, batch_y, mode='train'):
        if mode == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        x = batch_x.to(self.device).float()
        y = batch_y.to(self.device).float()

        outputs = self.model(x)
        loss_val = None
        
        if self.criterion:
            loss = self.criterion(outputs, y)
            loss_val = loss.item()
            if mode == 'train' and self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            preds = (outputs > self.threshold).float()
            # 기존 Trainer 로직: 하나라도 틀리면 에러
            error_count = (preds != y).any(dim=1).sum().item()

        return {
            "loss": loss_val,
            "error_count": error_count,
            "batch_size": y.size(0)
        }