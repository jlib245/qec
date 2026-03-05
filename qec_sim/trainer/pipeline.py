# qec_sim/trainer/pipeline.py

import os
import shutil
import datetime
import torch
import torch.optim as optim
from itertools import product
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
from qec_sim.config.schema import ExperimentConfig, NoiseParams
from qec_sim.data.generator import DatasetGenerator

class TrainingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path 
        # YAML 설정을 데이터 클래스 객체로 로드
        self.config = ExperimentConfig.from_yaml(config_path)
        # 하드웨어 장치 설정
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

    def _expand_noise_configs(self) -> list[NoiseParams]:
        """YAML의 리스트 형태 노이즈 설정을 모든 경우의 수(Cartesian Product)로 확장"""
        n = self.config.noise
        keys = ['p_gate', 'p_meas', 'p_corr', 'p_leak']
        
        # 각 속성이 리스트면 그대로 사용, 단일값이면 리스트로 래핑
        values = [getattr(n, k) if isinstance(getattr(n, k), list) else [getattr(n, k)] for k in keys]
        
        # itertools.product를 이용해 모든 조합 생성
        combinations = product(*values)
        
        return [
            NoiseParams(**dict(zip(keys, v))) 
            for v in combinations
        ]

    def _ensure_datasets_exist(self):
        """데이터가 없으면 DatasetGenerator를 호출하여 자동 생성 및 셔플링 수행"""
        train_path = Path(self.config.training.train_path)
        val_path = Path(self.config.training.val_path)

        if not train_path.exists():
            print(f"🔍 데이터를 찾을 수 없습니다. 자동 생성을 시작합니다: {train_path}")
            
            # 노이즈 조합 리스트 생성
            noise_configs = self._expand_noise_configs()
            
            # 기존 구현된 Generator 활용 (내부에 분산 생성 및 글로벌 셔플 로직 포함)
            generator = DatasetGenerator(self.config.code, noise_configs)
            
            # 훈련 데이터 생성 (chunk_size를 배치 단위로 사용)
            generator.generate_and_save(
                shots=self.config.training.train_steps * self.config.training.batch_size,
                save_dir=str(train_path.parent),
                filename=train_path.stem,
                batch_size=self.config.training.chunk_size
            )

            # 검증 데이터 생성
            generator.generate_and_save(
                shots=self.config.training.val_steps * self.config.training.batch_size,
                save_dir=str(val_path.parent),
                filename=val_path.stem,
                batch_size=self.config.training.chunk_size
            )

    def _get_optimizer(self, model):
        """YAML 설정을 기반으로 옵티마이저 생성"""
        opt_config = self.config.training.optimizer
        return getattr(optim, opt_config['name'])(model.parameters(), **opt_config['kwargs'])

    def run(self):
        """전체 학습 파이프라인 실행 로직"""
        # 1. 작업 공간 및 데이터 확인
        self._setup_workspace()
        if self.config.training.data_mode == 'offline':
            self._ensure_datasets_exist()

        # 2. 데이터 로더 준비
        data_strategy = OfflineDataStrategy(
            train_path=self.config.training.train_path,
            val_path=self.config.training.val_path,
            batch_size=self.config.training.batch_size
        )
        datamodule = QECDataModule(strategy=data_strategy)
        train_loader, val_loader = datamodule.get_loaders()
        
        # 3. 모델 및 Loss(Criterion) 준비
        model = build_model(
            self.config.model.name, 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables, 
            **self.config.model.kwargs
        ).to(self.device)

        # 레지스트리를 통한 Loss 함수 빌드
        criterion = build_criterion(
            self.config.training.criterion.get('name', 'bce_with_logits'),
            **self.config.training.criterion.get('kwargs', {})
        )
        evaluator = Evaluator(device=self.device, criterion=criterion)

        # 4. 콜백 설정
        callbacks = [
            CSVLogger(log_path=self.workspace["log"]),
            ModelCheckpoint(save_path=self.workspace["best_model"], monitor='val_loss'),
            EarlyStopping(patience=self.config.training.early_stopping_patience, monitor='val_loss')
        ]

        # 5. 트레이너 실행
        trainer = Trainer(
            model=model, 
            evaluator=evaluator, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=self._get_optimizer(model),
            callbacks=callbacks,
            train_steps=self.config.training.train_steps,
            val_steps=self.config.training.val_steps
        )
        
        trainer.fit(epochs=self.config.training.epochs)
        
        # 마지막 모델 상태 저장
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

    def _setup_decoder(self, circuit):
        decoder_kwargs = self.config.decoder.model_kwargs.copy()
        decoder_kwargs.update({
            'error_model': circuit.detector_error_model(decompose_errors=True),
            'num_detectors': circuit.num_detectors,
            'num_observables': circuit.num_observables,
            'weight_path': self.config.decoder.weight_path
        })
        return build_decoder(self.config.decoder.name, **decoder_kwargs)

    def run(self):
        circuit, simulator = self._setup_simulator()
        decoder = self._setup_decoder(circuit)
        
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