# qec_sim/core/callbacks.py
import os
import csv
import copy
import torch

class Callback:
    """콜백 기본 인터페이스"""
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs=None): pass
    def on_train_end(self, trainer): pass

class CSVLogger(Callback):
    """CSV 로그를 기록하는 콜백"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []
        
    def on_epoch_end(self, trainer, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch + 1
        self.history.append(logs)
        
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)

class ModelCheckpoint(Callback):
    """최상의 검증 손실을 기록한 모델을 저장하는 콜백"""
    def __init__(self, save_path, monitor='val_loss'):
        self.save_path = save_path
        self.monitor = monitor
        self.best_value = float('inf')
        self.best_weights = None

    def on_epoch_end(self, trainer, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None and current < self.best_value:
            self.best_value = current
            # 모델 가중치 복사 (CPU로 옮겨서 저장 메모리 최적화)
            self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            
    def on_train_end(self, trainer):
        if self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
            torch.save(self.best_weights, self.save_path)
            print(f"✨ [ModelCheckpoint] 최적 모델 가중치(Loss: {self.best_value:.4f})가 '{self.save_path}'에 저장되었습니다.")

class EarlyStopping(Callback):
    """개선이 없을 때 학습을 조기 종료하는 콜백"""
    def __init__(self, patience=0, monitor='val_loss'):
        self.patience = patience
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, logs=None):
        if self.patience <= 0:
            return
            
        current = logs.get(self.monitor)
        if current is not None:
            if current < self.best_value:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n🛑 [EarlyStopping] {self.patience} 에포크 동안 개선이 없어 조기 종료합니다.")
                    trainer.stop_training = True