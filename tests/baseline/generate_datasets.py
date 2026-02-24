from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.data.generator import DatasetGenerator

def main():
    # 1. 훈련용 설정값 (예: distance 5에서 누설 2%, 상관관계 0.5%)
    code_config = CodeParams(distance=5, rounds=5)
    noise_config = NoiseParams(
        p_gate=0.005, 
        p_meas=0.005, 
        p_corr=0.005, 
        p_leak=0.02
    )
    
    # 2. 데이터셋 생성기 초기화
    generator = DatasetGenerator(code_config, noise_config)
    
    # 3. 저장할 폴더 이름 지정
    save_dir = "datasets/d5_complex_noise"
    
    # 4. Train, Val, Test 데이터 생성 
    # (일단 테스트용으로 Train 10만 개, Val/Test 1만 개씩 생성해봅니다)
    generator.generate_and_save(shots=100000, save_dir=save_dir, filename="train")
    generator.generate_and_save(shots=10000, save_dir=save_dir, filename="val")
    generator.generate_and_save(shots=10000, save_dir=save_dir, filename="test")

if __name__ == "__main__":
    main()