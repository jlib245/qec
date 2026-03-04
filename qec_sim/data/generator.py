import os
import numpy as np
import random
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator

class DatasetGenerator:
    def __init__(self, code_params: CodeParams, noise_config_dict: dict):
        self.code_params = code_params
        # 딕셔너리 형태의 설정을 리스트로 정규화하여 저장
        self.noise_lists = {
            k: (v if isinstance(v, list) else [v]) 
            for k, v in noise_config_dict.items()
        }

    def generate_and_save(self, shots: int, save_dir: str, filename: str, chunk_size: int = 1000):
        print(f"[{filename}] 데이터 생성을 시작합니다. (총 {shots} 샷, 균등 분포 적용)")
        
        all_syndromes = []
        all_observables = []
        all_erasures = []

        generated_shots = 0
        while generated_shots < shots:
            current_batch = min(chunk_size, shots - generated_shots)
            
            # 매 청크마다 리스트에서 값을 하나씩 뽑아 NoiseParams 객체 생성
            sampled_noise_kwargs = {
                k: random.choice(v) for k, v in self.noise_lists.items()
            }
            noise_params = NoiseParams(**sampled_noise_kwargs)
            
            # 샘플링된 단일 값 객체로 빌더와 시뮬레이터 생성
            builder = CustomCircuitBuilder(self.code_params, noise_params)
            simulator = ComplexNoiseSimulator(builder.build(), noise_params)
            
            s, o, e = simulator.generate_data(shots=current_batch)
            
            all_syndromes.append(s)
            all_observables.append(o)
            all_erasures.append(e)
            
            generated_shots += current_batch
            if (generated_shots // chunk_size) % 10 == 0:
                print(f"진행률: {generated_shots}/{shots}")

        # 저장 로직 (기존과 동일)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.npz")
        np.savez_compressed(
            save_path,
            syndromes=np.concatenate(all_syndromes, axis=0),
            observables=np.concatenate(all_observables, axis=0),
            erasures=np.concatenate(all_erasures, axis=0)
        )
        print(f"✅ {filename} 저장 완료: {save_path}")