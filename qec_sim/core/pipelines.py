# qec_sim/core/pipelines.py

import yaml
import torch
import torch.optim as optim

# 필요한 모듈 임포트
from qec_sim.data import QECDataModule
from qec_sim.models import build_model
from qec_sim.core.trainer import QECTrainer
from qec_sim.core.evaluator import QECEvaluator
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator
from qec_sim.decoders import build_decoder

class TrainingPipeline:
    """YAML 설정 파일을 읽어 처음부터 끝까지 모델 학습을 진행하는 파이프라인"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.train_config = self.config.get('training', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[{config_path}] 학습 파이프라인 초기화 완료 (디바이스: {self.device})")

    def run(self):
        # 1. 데이터 준비
        datamodule = QECDataModule(self.config)
        train_loader, val_loader = datamodule.get_loaders()
        
        # 2. 모델 및 옵티마이저 준비
        model_config = self.config.get('model', {})
        model = build_model(
            model_config.get('name', 'erasure_mlp'), 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables, 
            **model_config.get('kwargs', {})
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.train_config.get('learning_rate', 0.001))
        
        # 3. 학습 엔진 구동
        trainer = QECTrainer(model, train_loader, val_loader, optimizer, self.device)
        trainer.fit(epochs=self.train_config.get('epochs', 20))
        trainer.save_model(save_path=self.train_config.get('save_path', 'model_weights.pth'))


class EvaluationPipeline:
    """YAML 설정 파일을 읽어 시뮬레이션 데이터 생성 및 디코딩 성능을 평가하는 파이프라인"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"[{config_path}] 평가 파이프라인 초기화 완료")

    def run(self):
        # 1. 양자 회로 및 시뮬레이터 준비
        code_config = CodeParams(**self.config['code'])
        noise_config = NoiseParams(**self.config['noise'])
        
        builder = CustomCircuitBuilder(code_config, noise_config)
        circuit = builder.build()
        error_model = circuit.detector_error_model(decompose_errors=True)
        simulator = ComplexNoiseSimulator(circuit, noise_config)

        # 2. 디코더 준비
        decoder_kwargs = self.config.get('decoder', {}).copy()
        decoder_name = decoder_kwargs.pop('name') 
        decoder_kwargs['error_model'] = error_model
        decoder_kwargs['num_detectors'] = circuit.num_detectors
        decoder_kwargs['num_observables'] = circuit.num_observables
        decoder = build_decoder(decoder_name, **decoder_kwargs)

        # 3. 평가 엔진 구동
        shots = self.config.get('simulation', {}).get('shots', 1000)
        evaluator = QECEvaluator(simulator, decoder)
        results = evaluator.evaluate(shots=shots)
        evaluator.print_results(results)