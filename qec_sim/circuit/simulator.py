# qec_sim/core/simulator.py

import stim
import numpy as np
import random

from qec_sim.config.schema import NoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.core.interfaces import BaseSimulator

class CircuitNoiseSimulator(BaseSimulator):
    def __init__(self, circuit: stim.Circuit, noise_params: NoiseParams):
        self._circuit = circuit
        self.noise = noise_params
        self._num_detectors = circuit.num_detectors
        self._num_observables = circuit.num_observables

    @property
    def num_detectors(self) -> int:
        return self._num_detectors

    @property
    def num_observables(self) -> int:
        return self._num_observables

    def generate_data(self, shots: int):
        """
        주어진 샷(shots) 수만큼 detection event 신드롬과 logical observable을 반환합니다.
        파울리 노이즈는 이미 회로에 반영되어 Stim sampler가 처리합니다.
        """
        stim_seed = int(np.random.randint(2 ** 31))
        sampler = self._circuit.compile_detector_sampler(seed=stim_seed)
        syndromes, observables = sampler.sample(
            shots=shots, separate_observables=True
        )
        return {
            'syndromes': syndromes,
            'observables': observables,
        }
    
class SimulatorPool:
    """여러 노이즈 환경의 시뮬레이터를 묶어서 관리하는 풀(Pool) 클래스"""

    def __init__(self, simulators: list):
        assert len(simulators) > 0
        self.simulators = simulators
        self.num_detectors = simulators[0].num_detectors
        self.num_observables = simulators[0].num_observables

    @classmethod
    def from_stim(cls, code_config, noise_configs) -> 'SimulatorPool':
        """기존 Stim 기반 시뮬레이터 풀 생성 (하위 호환용)"""
        noise_configs = noise_configs if isinstance(noise_configs, list) else [noise_configs]
        simulators = []
        for n_config in noise_configs:
            builder = build_circuit(code_config.name, code_config, n_config)
            simulators.append(CircuitNoiseSimulator(builder.build(), n_config))
        return cls(simulators)

    def get_random_simulator(self):
        return random.choice(self.simulators)

    def get_all_simulators(self) -> list:
        return self.simulators