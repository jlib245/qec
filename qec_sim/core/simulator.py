# qec_sim/core/simulator.py

import stim
import numpy as np
from .parameters import NoiseParams

class ComplexNoiseSimulator:
    def __init__(self, circuit: stim.Circuit, noise_params: NoiseParams):
        self.circuit = circuit
        self.noise = noise_params
        # Stim의 빠른 샘플러 컴파일
        self.sampler = circuit.compile_detector_sampler()

    def generate_data(self, shots: int):
        """
        주어진 샷(shots) 수만큼 데이터를 생성하고,
        누설(Leakage) 모델을 적용하여 최종 신드롬과 Erasure Mask를 반환합니다.
        """
        # 1. 베이스라인 샘플링 (파울리 노이즈 및 상관 관계 노이즈는 이미 회로에 반영됨)
        # syndromes: (shots, num_detectors) 크기의 boolean 배열
        syndromes, observables = self.sampler.sample(
            shots=shots, separate_observables=True
        )
        
        # 2. Erasure Mask 초기화 (기본은 누설 없음: False)
        erasure_masks = np.zeros_like(syndromes, dtype=bool)
        
        # 3. 누설(Leakage) 적용
        if self.noise.p_leak > 0:
            # 모델링: 특정 확률(p_leak)로 신드롬 측정 결과가 '소멸(Erasure)'되었다고 가정.
            # (실제로는 하드웨어의 특정 큐비트가 누설된 것이지만, 
            # 디코딩 관점에서는 해당 큐비트와 연결된 디텍터의 신뢰도가 0이 되는 것과 같습니다.)
            erasure_masks = np.random.random(syndromes.shape) < self.noise.p_leak
            
            # 누설이 발생한 위치의 측정값은 정보를 잃었으므로 50% 확률로 무작위(Random) 값이 됨
            random_noise = np.random.random(syndromes.shape) < 0.5
            
            # 누설된 부분(erasure_masks == True)의 신드롬만 무작위로 뒤집음(XOR)
            syndromes = np.where(erasure_masks, syndromes ^ random_noise, syndromes)

        # 최종적으로 3가지 데이터를 반환
        # 1) syndromes: 누설까지 반영된 최종 에러 신호
        # 2) observables: 우리가 지켜야 할 진짜 논리값 (정답 라벨)
        # 3) erasure_masks: 어디서 누설이 났는지 알려주는 1/0 맵 (딥러닝 모델의 추가 입력으로 사용!)
        return syndromes, observables, erasure_masks