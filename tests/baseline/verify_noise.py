# tests/baseline/verify_noise.py
import argparse
import yaml
import numpy as np

from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator

def verify_error_model(config_path):
    print(f"🔍 [{config_path}] 오류 모델 검증을 시작합니다...\n")

    # 1. 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    code_config = CodeParams(**config['code'])
    noise_config = NoiseParams(**config['noise'])
    shots = 1000000  

    print("=== 1. 주입된 노이즈 파라미터 (Expected) ===")
    print(f"- 게이트 에러 (p_gate): {noise_config.p_gate:.4f}")
    print(f"- 측정 에러 (p_meas): {noise_config.p_meas:.4f}")
    print(f"- 소실/누설 에러 (p_leak): {noise_config.p_leak:.4f}")
    print("-" * 40)

    # 2. 회로 생성
    builder = CustomCircuitBuilder(code_config, noise_config)
    circuit = builder.build()
    
    print("\n=== 2. Stim 양자 회로 정보 ===")
    print(f"- 총 게이트/명령어 수: {len(circuit)}")
    print(f"- 디텍터(Detector) 수: {circuit.num_detectors}")
    print(f"- 논리 옵저버블(Observable) 수: {circuit.num_observables}")
    
    # (선택) 회로의 앞부분 10줄만 출력하여 육안 확인
    
    print("- 회로 명령어 미리보기 (Top 10 lines):")
    circuit_str = str(circuit).split('\n')
    """
    for line in circuit_str[:10]:
        print(f"  {line}")
    print("  ...")
    print("-" * 40)
    """
    # 3. 데이터 샘플링 (디코더 없이 순수 데이터만 추출)
    simulator = ComplexNoiseSimulator(circuit, noise_config)
    syndromes, observables, erasures = simulator.generate_data(shots=shots)

    print("\n=== 3. 시뮬레이션 결과 통계 검증 (Actual) ===")
    print(f"- 테스트 샷(Shots) 수: {shots:,}")
    
    # 신드롬 통계 (디텍터가 에러를 감지한 비율)
    avg_syndromes_per_shot = np.mean(np.sum(syndromes, axis=1))
    syndrome_fraction = np.mean(syndromes) * 100
    print(f"- 샷 당 평균 신드롬 발생 횟수: {avg_syndromes_per_shot:.2f} 개")
    print(f"- 전체 디텍터 중 신드롬(1)이 켜질 확률: {syndrome_fraction:.2f}%")

    # Erasure 통계 (누설이 발생한 비율)
    if erasures is not None:
        avg_erasures_per_shot = np.mean(np.sum(erasures, axis=1))
        erasure_fraction = np.mean(erasures) * 100
        print(f"- 샷 당 평균 소실(Erasure) 발생 횟수: {avg_erasures_per_shot:.2f} 개")
        print(f"- 전체 위치 중 소실(1)로 마킹될 확률: {erasure_fraction:.2f}%")
        
        # p_leak 값과 실제 erasure_fraction이 비슷한지 비교
        print(f"\n💡 [진단] 설정된 p_leak: {noise_config.p_leak*100:.2f}%, "
              f"실제 소실 마킹 비율: {erasure_fraction:.2f}%")
    else:
        print("- 소실(Erasure) 데이터: 없음 (p_leak이 0이거나 비활성화됨)")

    # 옵저버블 통계 (아무런 정정을 하지 않았을 때의 원시 논리 에러율)
    raw_logical_error = np.mean(np.any(observables, axis=1)) * 100
    print(f"- [참고] 디코딩 전 원시(Raw) 논리적 에러율: {raw_logical_error:.2f}%")
    print("-" * 40)
    print("✅ 검증 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_mwpm.yaml")
    args = parser.parse_args()
    verify_error_model(args.config)