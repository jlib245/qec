# qec_sim/data/generator.py
import numpy as np
from pathlib import Path
import time

from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator

class DatasetGenerator:
    def __init__(self, code_config: CodeParams, noise_configs: list[NoiseParams]):
        self.code_config = code_config
        # 단일 설정이 들어와도 리스트로 래핑하여 통일
        self.noise_configs = noise_configs if isinstance(noise_configs, list) else [noise_configs]

    def generate_and_save(self, shots: int, save_dir: str, filename: str, batch_size: int = 50000):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / f"{filename}.npz"
        
        num_configs = len(self.noise_configs)
        print(f"[{filename}] 총 {shots}샷 생성을 시작합니다. ({num_configs}개의 노이즈 환경 분산)")
        start_time = time.time()
        
        # 메타데이터를 알기 위해 첫 번째 환경으로 임시 회로 빌드 (레지스트리 사용)
        temp_builder = build_circuit(self.code_config.name, self.code_config, self.noise_configs[0])
        temp_circuit = temp_builder.build()
        
        # 메모리 할당 
        final_syndromes = np.zeros((shots, temp_circuit.num_detectors), dtype=np.int8)
        final_observables = np.zeros((shots, temp_circuit.num_observables), dtype=np.int8)
        final_erasures = np.zeros((shots, temp_circuit.num_detectors), dtype=np.int8)
        
        # 전체 샷 N등분 계산 (나머지 처리 포함)
        base_shots = shots // num_configs
        remainder = shots % num_configs
        
        generated_count = 0
        
        for i, n_config in enumerate(self.noise_configs):
            config_shots = base_shots + (1 if i < remainder else 0)
            if config_shots == 0: continue
            
            print(f"  -> 환경 {i+1}/{num_configs} (p_gate:{n_config.p_gate:.4f}, p_leak:{n_config.p_leak:.4f} 등): {config_shots}샷 생성 중")
            
            # 레지스트리를 이용한 동적 빌더 및 시뮬레이터 생성
            builder = build_circuit(self.code_config.name, self.code_config, n_config)
            simulator = CircuitNoiseSimulator(builder.build(), n_config)
            
            config_generated = 0
            while config_generated < config_shots:
                current_shots = min(config_shots - config_generated, batch_size)
                syndromes, observables, erasures = simulator.generate_data(shots=current_shots)
                
                end_idx = generated_count + current_shots
                final_syndromes[generated_count:end_idx] = syndromes.astype(np.int8)
                final_observables[generated_count:end_idx] = observables.astype(np.int8)
                final_erasures[generated_count:end_idx] = erasures.astype(np.int8)
                
                generated_count = end_idx
                config_generated += current_shots
        
        # 딥러닝 데이터 편향을 막기 위한 전체 데이터 랜덤 셔플링
        print("  -> 데이터 혼합(Shuffling) 중...")
        indices = np.random.permutation(shots)
        
        np.savez_compressed(
            filepath,
            syndromes=final_syndromes[indices],
            observables=final_observables[indices],
            erasures=final_erasures[indices]
        )
        
        print(f"✅ 저장 완료: {filepath} (소요 시간: {time.time() - start_time:.2f}초)\n")