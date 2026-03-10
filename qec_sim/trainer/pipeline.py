import os
import datetime
import torch

from qec_sim.config.schema import ExperimentConfig
from qec_sim.trainer.factory import ComponentFactory
from qec_sim.trainer.trainer import Trainer
from qec_sim.metrics.evaluator import Evaluator
from qec_sim.metrics.registry import build_criterion
from qec_sim.trainer.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

class TrainingPipeline:
    """
    [최상위 계층: 실행 진입점]
    사용자의 Config를 읽어들여 시스템을 조립하고, 학습 루프를 끝까지 완주시키는 오케스트라 지휘자입니다.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ExperimentConfig.from_yaml(config_path)
        
        # Mac(MPS), NVIDIA(CUDA), CPU 자동 할당
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.workspace = {}

    def _setup_workspace(self):
        """결과물을 저장할 디렉토리와 파일 경로를 독립적으로 생성합니다."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.training.output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.workspace = {
            "root": output_dir,
            "log": os.path.join(output_dir, "training_log.csv"),
            "best_model": os.path.join(output_dir, "best_model.pth"),
            "last_model": os.path.join(output_dir, "last_model.pth")
        }

    def run(self):
        """파이프라인의 전체 실행 흐름"""
        print(f"🚀 학습 파이프라인 시작 (Device: {self.device})")
        self._setup_workspace()
        
        # 1. 부품 조립 
        # 팩토리가 Config를 보고 데이터 공급기와 래핑된 모델을 완벽히 짝지어 반환합니다.
        datamodule, wrapped_model = ComponentFactory.build_system(self.config)
        wrapped_model = wrapped_model.to(self.device)
        
        # 2. 데이터 준비 및 로더 획득
        print("데이터를 준비합니다...")
        datamodule.strategy.prepare() 
        train_loader, val_loader = datamodule.get_loaders()

        # 3. 손실 함수, 평가 지표, 옵티마이저, 스케줄러 세팅
        criterion = build_criterion(
            self.config.training.criterion['name'], 
            **self.config.training.criterion.get('kwargs', {})
        )
        evaluator = Evaluator(device=self.device, criterion=criterion)
        
        # 래퍼 모델의 파라미터를 옵티마이저에 전달 (내부 코어 모델의 파라미터도 자동 포함됨)
        optimizer = getattr(torch.optim, self.config.training.optimizer['name'])(
            wrapped_model.parameters(), 
            **self.config.training.optimizer['kwargs']
        )
        
        scheduler = None
        if hasattr(self.config.training, 'scheduler') and self.config.training.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.config.training.scheduler['name'])(
                optimizer, 
                **self.config.training.scheduler.get('kwargs', {})
            )

        # 4. 콜백 설정 (로깅, 체크포인트, 조기 종료)
        patience = self.config.training.early_stopping['patience']
        callbacks = [
            CSVLogger(log_path=self.workspace["log"]),
            ModelCheckpoint(save_path=self.workspace["best_model"], monitor='val_loss'),
            EarlyStopping(patience=patience, monitor='val_loss')
        ]

        # 5. 트레이너 실행
        print("트레이너를 초기화하고 학습을 시작합니다...")
        trainer = Trainer(
            wrapped_model=wrapped_model,
            evaluator=evaluator,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            train_steps=getattr(self.config.training, 'train_steps', None),
            val_steps=getattr(self.config.training, 'val_steps', None)
        )
        
        # 학습 루프 진입
        trainer.fit(epochs=self.config.training.epochs)
        
        # 학습 완전 종료 후 마지막 상태 저장
        torch.save(wrapped_model.state_dict(), self.workspace["last_model"])
        print(f"✅ 학습 파이프라인 종료. 모든 결과가 {self.workspace['root']} 디렉토리에 저장되었습니다.")