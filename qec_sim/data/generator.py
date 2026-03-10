# qec_sim/data/generator.py
import numpy as np
from pathlib import Path
import time

from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.simulator import SimulatorPool

class DatasetGenerator:
    def __init__(self, code_config: CodeParams, noise_configs: list[NoiseParams]):
        self.pool = SimulatorPool(code_config, noise_configs)

    def generate_and_save(self, shots: int, save_dir: str, filename: str, batch_size: int = 50000):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / f"{filename}.npz"
        
        simulators = self.pool.get_all_simulators()
        num_configs = len(simulators)
        print(f"[{filename}] 총 {shots}샷 생성을 시작합니다. ({num_configs}개의 노이즈 환경 분산)")
        start_time = time.time()
        
        # 첫 번째 시뮬레이터로 생성되는 키 확인 후 버퍼 초기화
        probe = simulators[0].generate_data(shots=1)
        label_key = 'observables'
        feature_keys = [k for k in probe if k != label_key]

        buffers = {label_key: np.zeros((shots, self.pool.num_observables), dtype=np.int8)}
        for k in feature_keys:
            buffers[k] = np.zeros((shots, probe[k].shape[1]), dtype=np.int8)

        base_shots = shots // num_configs
        remainder = shots % num_configs
        generated_count = 0

        for i, simulator in enumerate(simulators):
            config_shots = base_shots + (1 if i < remainder else 0)
            if config_shots == 0:
                continue

            print(f"  -> 환경 {i+1}/{num_configs} (p_gate:{simulator.noise.p_gate:.4f}, p_leak:{simulator.noise.p_leak:.4f} 등): {config_shots}샷 생성 중")

            config_generated = 0
            while config_generated < config_shots:
                current_shots = min(config_shots - config_generated, batch_size)
                raw = simulator.generate_data(shots=current_shots)

                end_idx = generated_count + current_shots
                for k in buffers:
                    buffers[k][generated_count:end_idx] = raw[k].astype(np.int8)

                generated_count = end_idx
                config_generated += current_shots

        print("  -> 데이터 혼합(Shuffling) 중...")
        indices = np.random.permutation(shots)

        np.savez_compressed(filepath, **{k: v[indices] for k, v in buffers.items()})
        
        print(f"✅ 저장 완료: {filepath} (소요 시간: {time.time() - start_time:.2f}초)\n")