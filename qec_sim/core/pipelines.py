# qec_sim/core/pipelines.py

import yaml
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import shutil
import datetime


from qec_sim.data import QECDataModule
from qec_sim.models import build_model
from qec_sim.core.trainer import QECTrainer
from qec_sim.core.evaluator import QECEvaluator
from qec_sim.core.parameters import CodeParams, NoiseParams, get_noise_combinations
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator
from qec_sim.decoders import build_decoder
from qec_sim.core.engine import QECEngine

class TrainingPipeline:
    """YAML 설정 파일을 읽어 처음부터 끝까지 모델 학습을 진행하는 파이프라인"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.train_config = self.config['training']
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[{config_path}] 학습 파이프라인 초기화 완료 (디바이스: {self.device})")

    def run(self):
        """학습 파이프라인 메인 실행 흐름 (자체 문서화)"""
        output_dir = self._setup_output_dir()
        train_loader, val_loader, datamodule = self._prepare_data()
        
        model = self._build_model(datamodule)
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)
        
        self._run_training(model, train_loader, val_loader, optimizer, scheduler, output_dir)

    def _setup_output_dir(self) -> str:
        base_output_dir = self.train_config['output_dir']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_output_dir}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(self.config_path, os.path.join(output_dir, "config.yaml"))
        
        print(f"📁 실험 결과 폴더 : {output_dir}")
        return output_dir

    def _prepare_data(self):
        datamodule = QECDataModule(self.config)
        train_loader, val_loader = datamodule.get_loaders()
        return train_loader, val_loader, datamodule

    def _build_model(self, datamodule):
        model_config = self.config['model']
        model = build_model(
            model_config['name'], 
            num_detectors=datamodule.num_detectors, 
            num_observables=datamodule.num_observables, 
            **model_config['kwargs']
        ).to(self.device)
        return model

    def _setup_optimizer(self, model):
        optim_config = self.train_config['optimizer']
        optim_name = optim_config['name']
        optim_kwargs = optim_config['kwargs']

        try:
            optimizer = getattr(optim, optim_name)(model.parameters(), **optim_kwargs)
            print(f"[{optim_name}] 옵티마이저가 성공적으로 로드되었습니다.")
            return optimizer
        except AttributeError:
            raise ValueError(f"지원하지 않는 옵티마이저입니다: {optim_name}")

    def _setup_scheduler(self, optimizer):
        sched_config = self.train_config['scheduler']
        if not sched_config:
            return None
            
        sched_name = sched_config['name']
        try:
            scheduler = getattr(lr_scheduler, sched_name)(optimizer, **sched_config['kwargs'])
            print(f"[{sched_name}] 스케줄러가 로드되었습니다.")
            return scheduler
        except AttributeError:
            raise ValueError(f"지원하지 않는 스케줄러입니다: {sched_name}")

    def _run_training(self, model, train_loader, val_loader, optimizer, scheduler, output_dir):
        es_patience = self.train_config['early_stopping']['patience']
        
        trainer = QECTrainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            device=self.device, 
            scheduler=scheduler, 
            early_stopping_patience=es_patience,
            log_path=os.path.join(output_dir, "training_log.csv")
        )
        trainer.fit(epochs=self.train_config['epochs'])
        trainer.save_model(save_path=os.path.join(output_dir, "best_model.pth"))


class EvaluationPipeline:
    def __init__(self, config_path, model):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # 평가는 Loss/Optimizer가 필요 없으므로 None 주입 (단, 채점 기준 확인용으로 criterion 타입만 맞춤)
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss() 
        self.engine = QECEngine(model, criterion=criterion)

    def run(self):
        noise_configs = get_noise_combinations(self.config['noise'])
        shots = self.config['simulation']['shots']
        results = []

        print(f"🧪 총 {len(noise_configs)}개 노이즈 환경 평가 시작 (RAM 기반)")
        
        for i, n_cfg in enumerate(noise_configs):
            # 1. 시뮬레이터 준비
            from qec_sim.core.builder import CustomCircuitBuilder
            from qec_sim.core.simulator import ComplexNoiseSimulator
            from qec_sim.core.parameters import CodeParams
            
            circuit = CustomCircuitBuilder(CodeParams(**self.config['code']), n_cfg).build()
            sim = ComplexNoiseSimulator(circuit, n_cfg)
            
            # 2. 데이터 생성 (RAM)
            syn, obs, _ = sim.generate_data(shots)
            
            # 3. 엔진으로 채점 (Batch 처리 권장하나 10만 개 통째로 가능하면 아래처럼)
            batch_x = torch.from_numpy(syn)
            batch_y = torch.from_numpy(obs)
            
            # 메모리 보호를 위해 10,000개씩 나눠서 엔진 돌리기
            res = self.engine.step(batch_x, batch_y, mode='eval')
            
            print(f"   [{i+1}/{len(noise_configs)}] p_gate={n_cfg.p_gate:.4f} | LER: {res['ler']:.4e}")
            results.append({"p_gate": n_cfg.p_gate, "ler": res['ler']})

        self._summary(results)

    def _summary(self, res_list):
        print("\n" + "="*40)
        print(f"{'Gate Error':>12} | {'Logical Error':>12}")
        for r in res_list:
            print(f"{r['p_gate']:12.4f} | {r['ler']:12.4e}")