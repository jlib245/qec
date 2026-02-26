# qec_sim/core/evaluator.py

import numpy as np

class QECEvaluator:
    """
    시뮬레이터와 디코더를 받아 논리적 에러율(LER)을 평가하는 모듈.
    """
    def __init__(self, simulator, decoder):
        self.simulator = simulator
        self.decoder = decoder

    def evaluate(self, shots: int) -> dict:
        """지정된 샷 수만큼 시뮬레이션을 돌리고 결과를 딕셔너리로 반환."""
        # 1. 데이터 샘플링
        syndromes, observables, erasures = self.simulator.generate_data(shots=shots)

        # 2. 기본 모드 평가 (누설 정보 없음)
        pred_standard = self.decoder.decode_batch(syndromes, erasures=None)
        errors_standard = np.sum(np.any(pred_standard != observables, axis=1))

        # 3. Erasure 인지 모드 평가 (누설 정보 활용)
        pred_erasure = self.decoder.decode_batch(syndromes, erasures=erasures)
        errors_erasure = np.sum(np.any(pred_erasure != observables, axis=1))

        return {
            "shots": shots,
            "standard_errors": errors_standard,
            "standard_ler": errors_standard / shots,
            "erasure_errors": errors_erasure,
            "erasure_ler": errors_erasure / shots,
        }

    def print_results(self, results: dict):
        """결과 딕셔너리를 출력."""
        shots = results["shots"]
        err_std, ler_std = results["standard_errors"], results["standard_ler"]
        err_era, ler_era = results["erasure_errors"], results["erasure_ler"]

        print("\n=== 논리적 에러율(Logical Error Rate) 결과 ===")
        print(f"기본 모드: {ler_std * 100:.2f}% ({err_std}/{shots})")
        print(f"Erasure 인지 모드: {ler_era * 100:.2f}% ({err_era}/{shots})")