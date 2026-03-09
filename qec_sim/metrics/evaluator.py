# qec_sim/metrics/evaluator.py

import torch
import torch.nn as nn
import numpy as np
from qec_sim.decoders.base import BaseDecoder

class Evaluator:
    def __init__(self, device, criterion=None):
        self.device = device
        self.criterion = criterion

    def validate_on_loader(self, model: nn.Module, loader, val_steps, preprocessor=None) -> tuple[float, float]:
        """[Trainer 용] PyTorch DataLoader를 이용한 검증"""
        if self.criterion is None:
            raise ValueError("validate_on_loader를 사용하려면 criterion이 주입되어야 합니다.")
            
        model.eval()
        total_loss, total_errors, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if val_steps is not None and step >= val_steps:
                    break
                    
                # DataLoader 반환값 패킹 (Erasure가 있는지 없는지에 따라 유연하게 대처)
                if len(batch) == 3:
                    batch_syn, batch_era, batch_y = batch
                    batch_era = batch_era.to(self.device).float()
                else:
                    batch_syn, batch_y = batch
                    batch_era = None
                    
                batch_syn = batch_syn.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                # [추가됨] 전처리기가 있으면 모델 통과 전에 변환!
                if preprocessor is not None:
                    batch_x = preprocessor.process(batch_syn, batch_era)
                else:
                    batch_x = batch_syn
                
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
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