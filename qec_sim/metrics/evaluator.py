# qec_sim/metrics/evaluator.py

import torch
import torch.nn as nn
import numpy as np
from qec_sim.decoders.base import BaseDecoder

class Evaluator:
    """
    학습 중의 Validation과 최종 시뮬레이션 Evaluation을 
    동일한 기준으로 평가하는 상태 없는(Stateless) 통합 평가기
    """
    def __init__(self, device, criterion=None):
        self.device = device
        self.criterion = criterion

    def validate_on_loader(self, model: nn.Module, loader, val_steps) -> tuple[float, float]:
        """[Trainer 용] PyTorch DataLoader를 이용한 검증 및 Loss/LER 계산"""
        if self.criterion is None:
            raise ValueError("validate_on_loader를 사용하려면 criterion이 주입되어야 합니다.")
            
        model.eval()
        total_loss, total_errors, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(loader):
                if val_steps is not None and step >= val_steps:
                    break
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # BCEWithLogitsLoss 기준: 0보다 크면 1(Error), 아니면 0(Normal)
                preds = (outputs > 0.0).float()
                batch_errors = (preds.bool() != batch_y.bool()).any(dim=1).sum().item()
                
                total_errors += batch_errors
                total_samples += batch_y.size(0)
                
        return total_loss / len(loader), total_errors / total_samples

    def evaluate_simulator(self, decoder: BaseDecoder, simulator, shots: int) -> dict:
        """[Pipeline 용] 시뮬레이터와 디코더를 이용한 최종 논리적 에러율(LER) 벤치마크"""
        # 1. 시뮬레이터에서 데이터 생성
        syndromes, observables, erasures = simulator.generate_data(shots=shots)

        # 2. Standard 모드 평가 (Erasure 정보 미사용)
        pred_std = decoder.decode_batch(syndromes, erasures=None)
        err_std = np.sum(np.any(pred_std != observables, axis=1))

        # 3. Erasure 인지 모드 평가 (Erasure 정보 활용)
        pred_era = decoder.decode_batch(syndromes, erasures=erasures)
        err_era = np.sum(np.any(pred_era != observables, axis=1))

        return {
            "shots": shots,
            "standard_ler": err_std / shots,
            "standard_errors": err_std,
            "erasure_ler": err_era / shots,
            "erasure_errors": err_era
        }