# qec_sim/core/builder.py

import stim
from .parameters import NoiseParams, CodeParams

class CustomCircuitBuilder:
    def __init__(self, code_params: CodeParams, noise_params: NoiseParams):
        self.code = code_params
        self.noise = noise_params

    def build(self) -> stim.Circuit:
        """
        Stim의 기본 회로를 베이스로 하되, 우리가 원하는 상관 관계 노이즈를 주입합니다.
        (추후 이 부분을 완전히 수동 좌표 기반으로 업그레이드하기 쉽게 분리해둠)
        """
        base_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.code.distance,
            rounds=self.code.rounds,
            after_clifford_depolarization=self.noise.p_gate,
            before_measure_flip_probability=self.noise.p_meas
        )

        if self.noise.p_corr > 0:
            return self._inject_correlated_noise(base_circuit)
        
        return base_circuit

    def _inject_correlated_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """2-Qubit 게이트(CX, CZ) 직후에 상관 관계 에러(CORRELATED_ERROR) 주입"""
        new_circuit = stim.Circuit()
        for inst in circuit:
            new_circuit.append(inst)
            if inst.name in ["CX", "CZ"]:
                targets = [t.value for t in inst.targets_copy()]
                for i in range(0, len(targets), 2):
                    q1, q2 = targets[i], targets[i+1]
                    
                    # 수정된 부분:
                    # 1. 인자 순서: "명령어", [타겟들], 확률
                    # 2. 타겟 지정: stim.target_x()를 사용하여 두 큐비트에 XX 에러가 발생함을 명시
                    new_circuit.append(
                        "CORRELATED_ERROR", 
                        [stim.target_x(q1), stim.target_x(q2)], 
                        self.noise.p_corr
                    )
        return new_circuit