# qec_sim/core/trainer.py
import torch
import torch.nn as nn
import os
import copy
import csv

class QECTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
                 scheduler=None, early_stopping_patience=0, log_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.best_model_weights = None
        self.patience_counter = 0
        
        # 로깅 경로 설정 (CSV와 TXT 자동 분리)
        self.log_path_csv = log_path
        self.history = []
        
        if log_path:
            base_name, _ = os.path.splitext(log_path)
            self.log_path_txt = f"{base_name}.txt"
            
            # 학습 시작 전 기존 TXT 파일을 비우고 헤더 작성
            os.makedirs(os.path.dirname(self.log_path_txt) or '.', exist_ok=True)
            with open(self.log_path_txt, 'w', encoding='utf-8') as f:
                f.write("=== QEC Training Log ===\n")
        else:
            self.log_path_txt = None

    # 에포크 학습 로직
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float() # BCE를 위해 float 타입 캐스팅
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    # ✨ 에포크 검증(Validation) 및 LER 계산 로직
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_errors = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # 예측값 변환 (BCEWithLogitsLoss를 쓰므로 0보다 크면 1, 아니면 0)
                preds = (outputs > 0.0).float()
                
                # 하나라도 틀리면 논리적 에러(Logical Error)로 간주
                batch_errors = (preds != batch_y).any(dim=1).sum().item()
                total_errors += batch_errors
                total_samples += batch_y.size(0)
                
        val_loss = total_loss / len(self.val_loader)
        val_ler = total_errors / total_samples if total_samples > 0 else 0.0
        
        return val_loss, val_ler

    def fit(self, epochs):
        start_msg = "\n학습을 시작합니다."
        print(start_msg)
        self._write_to_txt(start_msg)

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_ler = self.validate_epoch()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = (f"[Epoch {epoch+1:02d}/{epochs}] "
                       f"LR: {current_lr:.6f} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | "
                       f"Val LER: {val_ler * 100:.2f}%")
            print(log_msg)
            self._write_to_txt(log_msg)

            metrics = {
                'epoch': epoch + 1,
                'lr': current_lr,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ler': val_ler
            }
            self.history.append(metrics)
            self._save_log_to_csv()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience:
                    es_msg = f"\n[Early Stopping] {self.early_stopping_patience} 에포크 동안 검증 손실이 개선되지 않아 학습을 멈춥니다."
                    print(es_msg)
                    self._write_to_txt(es_msg)
                    break
        
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            restore_msg = f"\nBest Val Loss: {self.best_val_loss:.4f}."
            print(restore_msg)
            self._write_to_txt(restore_msg)

    def _write_to_txt(self, message):
        if self.log_path_txt:
            with open(self.log_path_txt, 'a', encoding='utf-8') as f:
                f.write(message + "\n")

    def _save_log_to_csv(self):
        if not self.log_path_csv:
            return
            
        with open(self.log_path_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['epoch', 'lr', 'train_loss', 'val_loss', 'val_ler']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"모델 가중치가 '{save_path}'에 저장되었습니다.")