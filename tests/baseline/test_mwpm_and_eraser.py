import numpy as np
from qec_sim.core.parameters import NoiseParams, CodeParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator
from qec_sim.decoders.mwpm import ErasureMWPM

def main():
    # 1. 설정값 (누설 확률을 3%로 꽤 높게 줘보겠습니다)
    code_config = CodeParams(distance=5, rounds=5)  # d=5로 증가
    noise_config = NoiseParams(
        p_gate=0.005, 
        p_meas=0.005, 
        p_corr=0.001, 
        p_leak=0.03   # 누설 3%
    )

    # 2. 회로 생성 및 에러 모델 추출
    print("1. 회로 생성 중...")
    builder = CustomCircuitBuilder(code_config, noise_config)
    circuit = builder.build()
    error_model = circuit.detector_error_model(decompose_errors=True)

    # 3. 데이터 샘플링 (통계를 위해 1000샷)
    shots = 1000
    print(f"\n2. 시뮬레이션 샘플링 중... (Shots: {shots})")
    simulator = ComplexNoiseSimulator(circuit, noise_config)
    syndromes, observables, erasures = simulator.generate_data(shots=shots)

    # 4. 디코딩 준비
    print("\n3. 디코딩 진행 중...")
    decoder = ErasureMWPM(error_model)

    # [실험 A] 디코더가 누설(Erasure) 정보를 모를 때 (일반 MWPM)
    pred_standard = decoder.decode_batch(syndromes, erasures=None)
    errors_standard = np.sum(np.any(pred_standard != observables, axis=1))
    
    # [실험 B] 디코더가 누설(Erasure) 정보를 알 때 (Erasure MWPM)
    pred_erasure = decoder.decode_batch(syndromes, erasures=erasures)
    errors_erasure = np.sum(np.any(pred_erasure != observables, axis=1))

    # 5. 결과 출력
    print("\n=== 논리적 에러율(Logical Error Rate) 비교 ===")
    print(f"일반 MWPM (누설 정보 무시): {errors_standard / shots * 100:.2f}% ({errors_standard}/{shots})")
    print(f"Erasure MWPM (누설 정보 활용): {errors_erasure / shots * 100:.2f}% ({errors_erasure}/{shots})")
    
    if errors_erasure < errors_standard:
        print("\n🎉 성공! Erasure 정보를 활용해 성능이 눈에 띄게 향상되었습니다.")

if __name__ == "__main__":
    main()