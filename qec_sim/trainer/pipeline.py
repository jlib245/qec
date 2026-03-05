import os
import shutil
import datetime
import torch
import torch.optim as optim
import yaml

# 📂 새로운 디렉토리 구조에 맞춘 임포트
from qec_sim.config.schema import ExperimentConfig
from qec_sim.data.datamodule import QECDataModule, OfflineDataStrategy
from qec_sim.models.registry import build_model
from qec_sim.decoders.registry import build_decoder
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.metrics.evaluator import Evaluator
from qec_sim.trainer.trainer import Trainer
from qec_sim.trainer.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

class TrainingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path 
        # get() 없이 Fail-Fast를 적용한 엄격한 파싱
        self.config = ExperimentConfig.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")

    def run(self):
        # 1. 실험 결과 폴더 및 설정값 백업
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.training.output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(self.config_path, os.path.join(output_dir, "config.yaml"))
        print(f"📁 실험 결과 폴더 : {output_dir}")

        # 2. 데이터 파이프라인 조립 (Strategy 패턴)
        if self.config.training.data_mode == 'offline':
            data_strategy = OfflineDataStrategy(
                train_path=self.config.training.train_path,
                val_path=self.config.training.val_path,
                batch_size=self.config.training.batch_size
            )
        else:
            raise NotImplementedError("현재는 offline 모드만 지원합니다.")
            
        datamodule = QECDataModule(strategy=data_strategy)
        train_loader, val_loader = datamodule.get_loaders()
        
        # 3. 모델 준비
        model = build_model(
            self.config.model.name, 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables, 
            **self.config.model.kwargs
        ).to(self.device)

        # 4. 평가 계산기(Evaluator) 및 콜백 설정
        # 명시적으로 BCEWithLogitsLoss 주입 (기본값 의존 X)
        evaluator = Evaluator(device=self.device, criterion=torch.nn.BCEWithLogitsLoss())
        
        log_path_csv = os.path.join(output_dir, "training_log.csv")
        save_path_pth = os.path.join(output_dir, self.config.training.save_path)

        callbacks = [
            CSVLogger(log_path=log_path_csv),
            ModelCheckpoint(save_path=save_path_pth, monitor='val_loss'),
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
            train_steps=self.config.training.train_steps, # 1epoch당 스텝 수 제한 (옵션)
            val_steps=self.config.training.val_steps # 1epoch당 검증 스텝 수 제한 (옵션)
        )
        trainer.fit(epochs=self.config.training.epochs)

    def _get_optimizer(self, model):
        opt_config = self.config.training.optimizer
        return getattr(optim, opt_config['name'])(model.parameters(), **opt_config['kwargs'])


class EvaluationPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ExperimentConfig.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")

    def run(self):
        # 1. 양자 회로 및 시뮬레이터 준비 (명확해진 네이밍)
        builder = build_circuit(
            name=self.config.code.name, # 예: "surface_code" 또는 "color_code"
            code_config=self.config.code, 
            noise_config=self.config.noise
        )
        circuit = builder.build()
        simulator = CircuitNoiseSimulator(circuit, self.config.noise)

        # 2. 디코더 준비
        decoder_kwargs = self.config.decoder.model_kwargs.copy()
        decoder_kwargs.update({
            'error_model': circuit.detector_error_model(decompose_errors=True),
            'num_detectors': circuit.num_detectors,
            'num_observables': circuit.num_observables,
            'weight_path': self.config.decoder.weight_path
        })
        decoder = build_decoder(self.config.decoder.name, **decoder_kwargs)

        # 3. 엔진을 통한 평가 수행
        evaluator = Evaluator(device=self.device)
        # simulation['shots']가 YAML에 없으면 에러 발생 (Fail-Fast)
        results = evaluator.evaluate_simulator(
            decoder=decoder,
            simulator=simulator, 
            shots=self.config.simulation['shots'] 
        )

        print("\n=== 논리적 에러율(Logical Error Rate) 결과 ===")
        print(f"기본 모드: {results['standard_ler'] * 100:.2f}% ({results['standard_errors']}/{results['shots']})")
        print(f"Erasure 인지 모드: {results['erasure_ler'] * 100:.2f}% ({results['erasure_errors']}/{results['shots']})")