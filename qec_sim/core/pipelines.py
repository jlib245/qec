# qec_sim/core/pipelines.py

import yaml
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import shutil
import datetime

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
        self.config_path = config_path # 원본 설정 파일 경로 저장
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.train_config = self.config.get('training', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[{config_path}] 학습 파이프라인 초기화 완료 (디바이스: {self.device})")

    def run(self):
        # 0. 실험 결과 및 설정값 백업 (자동 타임스탬프 폴더 생성)
        base_output_dir = self.train_config.get('output_dir', 'results/default_run')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_output_dir}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        backup_config_path = os.path.join(output_dir, "config.yaml")
        shutil.copy(self.config_path, backup_config_path)
        
        print(f"📁 실험 결과 폴더 : {output_dir}")
        print(f"📝 설정값 백업 : {backup_config_path}")

        # 1. 데이터 준비
        datamodule = QECDataModule(self.config)
        train_loader, val_loader = datamodule.get_loaders()
        # datamodule 안에 있는 circuit에서 좌표를 추출하여 모델에 전달 (모델이 좌표를 필요로 하는 경우)
        # datamodule 구조에 따라 .circuit 접근 방식 확인 필요
        try:
            # YAML 설정값을 바탕으로 임시 회로를 생성 -> 디텍터 좌표.
            code_config = CodeParams(**self.config.get('code', {}))
            noise_config = NoiseParams(**self.config.get('noise', {}))
            temp_circuit = CustomCircuitBuilder(code_config, noise_config).build()
            
            detector_coords = temp_circuit.get_detector_coordinates()
        except Exception as e:
            print(f"경고: 좌표계를 생성할 수 없습니다. ({e})")
            detector_coords = None
            
        code_distance = self.config.get('code', {}).get('distance', 5)

        # 2. 모델 및 옵티마이저 준비
        model_config = self.config.get('model', {})
        yaml_kwargs = model_config.get('kwargs', {})
        
        model = build_model(
            model_config.get('name', 'erasure_mlp'), 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables,
            detector_coords=detector_coords,  
            code_distance=code_distance,      
            **yaml_kwargs                     # yaml에 명시된 설정값 (우선순위 높음)
        ).to(self.device)
        
        optim_config = self.train_config.get('optimizer', {})
        optim_name = optim_config.get('name', 'Adam') # 기본값 Adam
        optim_kwargs = optim_config.get('kwargs', {'lr': 0.001})

        try:
            OptimizerClass = getattr(optim, optim_name)
            optimizer = OptimizerClass(model.parameters(), **optim_kwargs)
            print(f"[{optim_name}] 옵티마이저가 성공적으로 로드되었습니다. (설정: {optim_kwargs})")
        except AttributeError:
            raise ValueError(f"지원하지 않는 옵티마이저입니다: {optim_name}")
        
        # 3. 스케줄러 설정
        sched_config = self.train_config.get('scheduler', {})
        scheduler = None
        if sched_config:
            sched_name = sched_config.get('name')
            sched_kwargs = sched_config.get('kwargs', {})
            try:
                SchedulerClass = getattr(lr_scheduler, sched_name)
                scheduler = SchedulerClass(optimizer, **sched_kwargs)
                print(f"[{sched_name}] 스케줄러가 로드되었습니다.")
            except AttributeError:
                raise ValueError(f"지원하지 않는 스케줄러입니다: {sched_name}")

        # 4. Early Stopping 설정
        es_config = self.train_config.get('early_stopping', {})
        es_patience = es_config.get('patience', 0) # 0이면 사용 안 함

        # ---------------------------------------------------------
        # 5. 로그 및 모델 저장 경로 설정 
        log_path_csv = os.path.join(output_dir, "training_log.csv")
        save_path_pth = os.path.join(output_dir, "best_model.pth")
        # ---------------------------------------------------------

        # 6. 학습 엔진 구동 (생성된 경로 주입)
        trainer = QECTrainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            device=self.device, 
            scheduler=scheduler, 
            early_stopping_patience=es_patience,
            log_path=log_path_csv  
        )
        
        trainer.fit(epochs=self.train_config.get('epochs', 20))
        
        trainer.save_model(save_path=save_path_pth) 


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