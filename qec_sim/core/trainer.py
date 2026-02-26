# qec_sim/core/trainer.py
import torch
import torch.nn as nn
import os

class QECTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        steps = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            steps += batch_x.size(0)
            
        return train_loss / steps

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                predictions = (outputs > 0).float()
                correct_predictions += (predictions == batch_y).all(dim=1).sum().item()
                val_steps += batch_x.size(0)
                
        val_loss /= val_steps
        logical_error_rate = 1.0 - (correct_predictions / val_steps)
        return val_loss, logical_error_rate

    def fit(self, epochs):
        print("\n🚀 본격적인 학습을 시작합니다!")
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_ler = self.validate_epoch()
            
            print(f"[Epoch {epoch+1:02d}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Logical Error Rate: {val_ler * 100:.2f}%")

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"\n✅ 학습 완료! 모델 가중치가 '{save_path}'에 저장되었습니다.")