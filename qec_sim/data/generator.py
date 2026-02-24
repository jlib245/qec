import numpy as np
from pathlib import Path
import time
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator

class DatasetGenerator:
    def __init__(self, code_config: CodeParams, noise_config: NoiseParams):
        self.code_config = code_config
        self.noise_config = noise_config
        
        # 회로와 시뮬레이터 초기화
        builder = CustomCircuitBuilder(code_config, noise_config)
        self.circuit = builder.build()
        self.simulator = ComplexNoiseSimulator(self.circuit, noise_config)

    def generate_and_save(self, shots: int, save_dir: str, filename: str, batch_size: int = 50000):
        """
        데이터를 생성하고 .npz 형식으로 압축 저장합니다.
        대용량 생성을 위해 batch_size 단위로 끊어서 처리합니다.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / f"{filename}.npz"
        
        print(f"[{filename}] 데이터 생성을 시작합니다. (총 {shots} 샷)")
        start_time = time.time()
        
        all_syndromes = []
        all_observables = []
        all_erasures = []
        
        shots_remaining = shots
        while shots_remaining > 0:
            current_shots = min(shots_remaining, batch_size)
            syndromes, observables, erasures = self.simulator.generate_data(shots=current_shots)
            
            # boolean 타입을 int8(0과 1)로 변환하여 저장 용량 최적화
            all_syndromes.append(syndromes.astype(np.int8))
            all_observables.append(observables.astype(np.int8))
            all_erasures.append(erasures.astype(np.int8))
            
            shots_remaining -= current_shots
            print(f"  -> 진행 상황: {shots - shots_remaining}/{shots} 샷 완료")
            
        # 리스트에 모인 배열들을 하나로 병합
        final_syndromes = np.concatenate(all_syndromes, axis=0)
        final_observables = np.concatenate(all_observables, axis=0)
        final_erasures = np.concatenate(all_erasures, axis=0)
        
        # npz 압축 포맷으로 저장 (PyTorch DataLoader에서 읽기 아주 좋음)
        np.savez_compressed(
            filepath,
            syndromes=final_syndromes,
            observables=final_observables,
            erasures=final_erasures
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 저장 완료: {filepath} (소요 시간: {elapsed:.2f}초)\n")