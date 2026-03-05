# qec_sim/trainer/pipeline.py

import os
import shutil
import datetime
import torch
import torch.optim as optim
from pathlib import Path

from qec_sim.config.schema import ExperimentConfig
from qec_sim.data.datamodule import QECDataModule, OfflineDataStrategy
from qec_sim.models.registry import build_model
from qec_sim.decoders.registry import build_decoder
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.metrics.evaluator import Evaluator
from qec_sim.metrics.registry import build_criterion
from qec_sim.trainer.trainer import Trainer
from qec_sim.trainer.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from qec_sim.data.generator import DatasetGenerator

class DataPreparer:
    """데이터셋의 존재 여부를 확인하고 없으면 자동 생성하는 단일 책임을 가집니다."""
    @staticmethod
    def prepare_offline_data(config: ExperimentConfig):
        train_path = Path(config.training.train_path)
        val_path = Path(config.training.val_path)

        if not train_path.exists():
            print(f"훈련 데이터를 찾을 수 없습니다. 자동 생성을 시작합니다: {train_path}")
            
            # schema.py로 넘긴 확장 로직을 호출
            noise_configs = config.get_expanded_noise_configs()
            generator = DatasetGenerator(config.code, noise_configs)
            
            # 훈련 데이터 생성
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

    def run(self):
        """전체 학습 파이프라인 실행 로직"""
        # 1. 작업 공간 생성
        self._setup_workspace()
        
        # 2. 데이터 준비 (DataPreparer Class 호출)
        if self.config.training.data_mode == 'offline':
            DataPreparer.prepare_offline_data(self.config)

        # 3. 데이터 로더 준비
        data_strategy = OfflineDataStrategy(
            train_path=self.config.training.train_path,
            val_path=self.config.training.val_path,
            batch_size=self.config.training.batch_size
        )
        datamodule = QECDataModule(strategy=data_strategy)
        train_loader, val_loader = datamodule.get_loaders()

        # 모델에 좌표를 전달하기 위해 기본 회로를 빌드하여 정보 추출
        noise_configs = self.config.get_expanded_noise_configs()
        builder = build_circuit(self.config.code.name, self.config.code, noise_configs[0])
        circuit = builder.build()
        
        # 4. 모델 및 Loss(Criterion) 준비
        model = build_model(
            self.config.model.name, 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables,
            detector_coords=circuit.get_detector_coordinates(),  # 좌표 정보 
            code_distance=self.config.code.distance,  # 코드 거리 
            **self.config.model.kwargs
        ).to(self.device)

        criterion = build_criterion(
            self.config.training.criterion['name'],
            **self.config.training.criterion.get('kwargs', {})
        )
        evaluator = Evaluator(device=self.device, criterion=criterion)

        # 조기 종료 patience 값 추출
        patience = self.config.training.early_stopping['patience']

        # 5. 콜백 설정
        callbacks = [
            CSVLogger(log_path=self.workspace["log"]),
            ModelCheckpoint(save_path=self.workspace["best_model"], monitor='val_loss'),
            EarlyStopping(patience=patience, monitor='val_loss')
        ]

        # 6. 트레이너 실행
        trainer = Trainer(
            model=model, 
            evaluator=evaluator, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=self._get_optimizer(model),
            scheduler=None, # 필요한 경우 스케줄러 연동  *****필요한 경우가 필수 아닌가?
            callbacks=callbacks,
            train_steps=self.config.training.train_steps,
            val_steps=self.config.training.val_steps
        )
        
        trainer.fit(epochs=self.config.training.epochs)
        
        torch.save(model.state_dict(), self.workspace["last_model"])
        print(f"✅ 학습 완료. 결과가 {self.workspace['root']}에 저장되었습니다.")

class EvaluationPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ExperimentConfig.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_simulator(self):
        builder = build_circuit(
            name=self.config.code.name,
            code_config=self.config.code, 
            noise_config=self.config.noise
        )
        circuit = builder.build()
        return circuit, CircuitNoiseSimulator(circuit, self.config.noise)

    def _setup_model(self, circuit) -> torch.nn.Module:
        """신경망 디코더를 위한 모델 준비 책임을 파이프라인으로 가져옵니다."""
        model = build_model(
            self.config.model.name,
            num_detectors=circuit.num_detectors,
            num_observables=circuit.num_observables,
            detector_coords=circuit.get_detector_coordinates(),
            code_distance=self.config.code.distance,
            **self.config.model.kwargs
        ).to(self.device)
        
        # 가중치 파일 로드
        if self.config.decoder.weight_path:
            model.load_state_dict(
                torch.load(self.config.decoder.weight_path, map_location=self.device)
            )
        return model

    def _setup_decoder(self, circuit, neural_model=None):
        decoder_kwargs = self.config.decoder.model_kwargs.copy()
        
        # MWPM 디코더용 에러 모델
        decoder_kwargs['error_model'] = circuit.detector_error_model(decompose_errors=True)
        
        # 신경망 디코더용 파이토치 모델 주입 (DI)
        if neural_model is not None:
            decoder_kwargs['model'] = neural_model
            
        return build_decoder(self.config.decoder.name, **decoder_kwargs)

    def run(self):
        circuit, simulator = self._setup_simulator()
        
        # 설정에 따라 필요한 경우에만 딥러닝 모델 초기화
        neural_model = None
        if self.config.decoder.name == "neural_decoder":
            neural_model = self._setup_model(circuit)
            
        # 디코더 생성 (이때 neural_model이 있다면 함께 전달됨)
        decoder = self._setup_decoder(circuit, neural_model=neural_model)
        
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