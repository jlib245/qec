import yaml
import argparse
import numpy as np

from qec_sim.core.parameters import NoiseParams, CodeParams
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator

# 레지스트리 임포트
from qec_sim.decoders import build_decoder
from qec_sim.models import build_model 

def main(config_path):
    # 1. YAML 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[{config_path}] 설정으로 실험을 시작합니다...")

    # 2. 파라미터 객체화
    code_config = CodeParams(**config['code'])
    noise_config = NoiseParams(**config['noise'])
    shots = config['simulation']['shots']

    # 3. 회로 생성 및 에러 모델 추출
    builder = CustomCircuitBuilder(code_config, noise_config)
    circuit = builder.build()
    error_model = circuit.detector_error_model(decompose_errors=True)

    # 4. 데이터 샘플링
    simulator = ComplexNoiseSimulator(circuit, noise_config)
    syndromes, observables, erasures = simulator.generate_data(shots=shots)

    # 5. 레지스트리에서 설정된 디코더 동적 생성! 
    decoder_kwargs = config.get('decoder', {}).copy()
    decoder_name = decoder_kwargs.pop('name') # 'name'만 빼고 나머지는 kwargs로 전달
    
    # 런타임에 알아내는 필수 정보들 주입 (MWPM은 error_model을, 신경망은 detector 수를 사용)
    decoder_kwargs['error_model'] = error_model
    decoder_kwargs['num_detectors'] = circuit.num_detectors
    decoder_kwargs['num_observables'] = circuit.num_observables
    
    decoder = build_decoder(decoder_name, **decoder_kwargs)

    print(f"선택된 디코더: {decoder_name}")

    # 6. 디코딩 실행 (BaseDecoder 인터페이스를 따르므로 동일한 방식 사용)
    pred_standard = decoder.decode_batch(syndromes, erasures=None)
    errors_standard = np.sum(np.any(pred_standard != observables, axis=1))
    
    pred_erasure = decoder.decode_batch(syndromes, erasures=erasures)
    errors_erasure = np.sum(np.any(pred_erasure != observables, axis=1))

    # 7. 결과 출력
    print("\n=== 논리적 에러율(Logical Error Rate) 결과 ===")
    print(f"기본 모드: {errors_standard / shots * 100:.2f}% ({errors_standard}/{shots})")
    print(f"Erasure 인지 모드: {errors_erasure / shots * 100:.2f}% ({errors_erasure}/{shots})")

if __name__ == "__main__":
    # 터미널에서 설정 파일을 인자로 받을 수 있도록 argparse 사용
    parser = argparse.ArgumentParser(description="QEC Simulator Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/experiment_mwpm.yaml", help="YAML 설정 파일 경로")
    args = parser.parse_args()
    
    main(args.config)