# qec_sim/core/trainer.py
import os
import copy
from qec_sim.core.logger import QECLogger

class QECTrainer:
    def __init__(self, engine, train_loader, val_loader, scheduler=None, 
                 early_stopping_patience=10, log_path=None):
        self.engine = engine
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.logger = QECLogger(log_path)
        
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.best_model_weights = None
        self.patience_counter = 0

    def fit(self, epochs):
        self.logger.info("학습 시작")
        for epoch in range(epochs):
            train_loss = self._run_epoch(self.train_loader, mode='train')
            val_loss, val_ler = self._run_epoch(self.val_loader, mode='eval', return_ler=True)
            
            lr = self.engine.optimizer.param_groups[0]['lr']
            self._log(epoch, epochs, lr, train_loss, val_loss, val_ler)
            
            if self.scheduler: self.scheduler.step(val_loss)
            if self._early_stopping(val_loss): break
            
        self._restore_best()

    def _run_epoch(self, loader, mode='train', return_ler=False):
        total_loss, total_err, total_samples = 0, 0, 0
        for x, y in loader:
            res = self.engine.step(x, y, mode=mode)
            total_loss += res['loss']
            total_err += res['error_count']
            total_samples += res['batch_size']
        
        avg_loss = total_loss / len(loader)
        if return_ler: return avg_loss, total_err / total_samples
        return avg_loss

    def _log(self, e, total_e, lr, t_loss, v_loss, v_ler):
        msg = f"[Epoch {e+1}/{total_e}] LR: {lr:.6f} | Train Loss: {t_loss:.4f} | Val LER: {v_ler*100:.2f}%"
        self.logger.info(msg)
        self.logger.log_metrics({'epoch': e+1, 'val_ler': v_ler, 'val_loss': v_loss})

    def _early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_weights = copy.deepcopy(self.engine.model.state_dict())
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience

    def _restore_best(self):
        if self.best_model_weights:
            self.engine.model.load_state_dict(self.best_model_weights)
            self.logger.info(f"최적 모델 (Best Loss: {self.best_val_loss:.4f})")