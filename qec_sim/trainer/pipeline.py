# qec_sim/trainer/pipeline.py

import os
import shutil
import datetime
import torch
import torch.optim as optim
from pathlib import Path

from qec_sim.config.schema import ExperimentConfig
from qec_sim.data.datamodule import QECDataModule, OfflineDataStrategy, OnlineDataStrategy
from qec_sim.data.generator import DatasetGenerator
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.metrics.evaluator import Evaluator
from qec_sim.metrics.registry import build_criterion
from qec_sim.trainer.trainer import Trainer
from qec_sim.trainer.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from qec_sim.builder import ComponentBuilder


class DataPreparer:
    """데이터셋의 존재 여부를 확인하고 없으면 자동 생성."""
    @staticmethod
    def prepare_offline_data(config: ExperimentConfig):
        train_path = Path(config.training.train_path)
        val_path = Path(config.training.val_path)

        if not train_path.exists():
            print(f"훈련 데이터를 찾을 수 없습니다. 자동 생성을 시작합니다: {train_path}")
            noise_configs = config.get_expanded_noise_configs()
            generator = DatasetGenerator(config.code, noise_configs)
            
            generator.generate_and_save(
                shots=config.training.train_steps * config.training.batch_size,
                save_dir=str(train_path.parent),
                filename=train_path.stem,
                batch_size=config.training.chunk_size
            )
            
        if not val_path.exists():
            print(f"검증 데이터를 찾을 수 없습니다. 자동 생성을 시작합니다: {val_path}")
            # 검증 데이터 생성
            generator.generate_and_save(
                shots=config.training.val_steps * config.training.batch_size,
                save_dir=str(val_path.parent),
                filename=val_path.stem,
                batch_size=config.training.chunk_size
            )


class TrainingPipeline:
    """모델 학습 파이프라인 (Orchestrator)"""
    def __init__(self, config_path: str):
        self.config_path = config_path 
        self.config = ExperimentConfig.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.workspace = {}

    def _setup_workspace(self):
        """실험 결과 폴더 생성 및 설정 파일 백업"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.training.output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.workspace = {
            "root": output_dir,
            "log": os.path.join(output_dir, "training_log.csv"),
            "best_model": os.path.join(output_dir, "best_model.pth"),
            "last_model": os.path.join(output_dir, "last_model.pth"),
            "config_backup": os.path.join(output_dir, "config.yaml")
        }
        
        shutil.copy(self.config_path, self.workspace["config_backup"])
        print(f"📁 결과 저장 위치: {output_dir}")

    def _get_optimizer(self, model):
        """YAML 설정을 기반으로 옵티마이저 생성"""
        opt_config = self.config.training.optimizer
        return getattr(optim, opt_config['name'])(model.parameters(), **opt_config['kwargs'])
    
    def _get_scheduler(self, optimizer):
        """YAML 설정을 기반으로 러닝레이트 스케줄러 생성"""
        sched_config = self.config.training.scheduler
        return getattr(torch.optim.lr_scheduler, sched_config['name'])(optimizer, **sched_config.get('kwargs', {}))

    def run(self):
        """전체 학습 파이프라인 실행 로직"""
        # --- [단계 1] 작업 공간 셋업 ---
        self._setup_workspace()
        
        # --- [단계 2] 데이터 준비 및 로더 생성 ---
        if self.config.training.data_mode == 'offline':
            DataPreparer.prepare_offline_data(self.config)
            data_strategy = OfflineDataStrategy(config=self.config)
        elif self.config.training.data_mode == 'online':
            data_strategy = OnlineDataStrategy(config=self.config)
        else:
            raise ValueError(f"지원하지 않는 데이터 모드입니다.: {self.config.training.data_mode}")
        
        datamodule = QECDataModule(strategy=data_strategy)
        train_loader, val_loader = datamodule.get_loaders()

        # --- [단계 3] 도메인 환경 파악 (회로 빌드) ---
        # 실제 모델에 주입해야 할 메타데이터(좌표, 디텍터 수 등)를 추출하기 위해 임시로 서킷을 생성합니다.
        noise_configs = self.config.get_expanded_noise_configs()
        builder = build_circuit(self.config.code.name, self.config.code, noise_configs[0])
        circuit = builder.build()
        
        # --- [단계 4] 부품 조립 (Builder 위임) ---
        preprocessor, model = ComponentBuilder.build_neural_components(
            config=self.config,
            circuit=circuit,
            num_detectors=datamodule.num_detectors,
            device=self.device
        )

        # --- [단계 5] Loss, Evaluator, Callbacks 준비 ---
        criterion = build_criterion(
            self.config.training.criterion['name'],
            **self.config.training.criterion.get('kwargs', {})
        )
        evaluator = Evaluator(device=self.device, criterion=criterion)

        patience = self.config.training.early_stopping['patience']
        callbacks = [
            CSVLogger(log_path=self.workspace["log"]),
            ModelCheckpoint(save_path=self.workspace["best_model"], monitor='val_loss'),
            EarlyStopping(patience=patience, monitor='val_loss')
        ]

        # --- [단계 6] 트레이너 실행 ---
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        
        trainer = Trainer(
            model=model, 
            evaluator=evaluator, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            train_steps=self.config.training.train_steps,
            val_steps=self.config.training.val_steps,
            preprocessor=preprocessor # (선택) Dataset/Dataloader에서 전처리기를 활용할 수 있도록 주입
        )
        
        trainer.fit(epochs=self.config.training.epochs)
        
        torch.save(model.state_dict(), self.workspace["last_model"])
        print(f"✅ 학습 완료. 결과가 {self.workspace['root']}에 저장되었습니다.")


class EvaluationPipeline:
    """시뮬레이션 및 에러율(LER) 평가 파이프라인 (Orchestrator)"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ExperimentConfig.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_simulator(self):
        """평가를 위한 Stim 서킷과 시뮬레이터 준비"""
        builder = build_circuit(
            name=self.config.code.name,
            code_config=self.config.code, 
            noise_config=self.config.noise
        )
        circuit = builder.build()
        return circuit, CircuitNoiseSimulator(circuit, self.config.noise)

    def run(self):
        """전체 평가(시뮬레이션) 파이프라인 실행 로직"""
        # --- [단계 1] 시뮬레이터 준비 ---
        circuit, simulator = self._setup_simulator()
        
        neural_model = None
        preprocessor = None
        
        # --- [단계 2] 신경망 모델 및 전처리기 조립 ---
        if self.config.decoder.name == "neural_decoder":
            preprocessor, neural_model = ComponentBuilder.build_neural_components(
                config=self.config,
                circuit=circuit,
                num_detectors=circuit.num_detectors,
                device=self.device
            )
            
            # 사전에 학습된 가중치(Weight) 로드
            if self.config.decoder.weight_path:
                neural_model.load_state_dict(
                    torch.load(self.config.decoder.weight_path, map_location=self.device)
                )
                
        # --- [단계 3] 최종 디코더 조립 ---
        decoder = ComponentBuilder.build_evaluation_decoder(
            config=self.config,
            circuit=circuit,
            neural_model=neural_model,
            preprocessor=preprocessor
        )
        
        # --- [단계 4] 평가 및 결과 출력 ---
        evaluator = Evaluator(device=self.device)
        results = evaluator.evaluate_simulator(
            decoder=decoder,
            simulator=simulator, 
            shots=self.config.simulation['shots'] 
        )

        self._print_results(results)

    def _print_results(self, results):
        print("\n=== 논리적 에러율(Logical Error Rate) 결과 ===")
        print(f"기본 모드: {results['standard_ler'] * 100:.2f}% ({results['standard_errors']}/{results['shots']})")
        print(f"Erasure 인지 모드: {results['erasure_ler'] * 100:.2f}% ({results['erasure_errors']}/{results['shots']})")