import time
from torch.utils.data import DataLoader
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.data.dataset import OfflineQECDataset, OnlineQECDataset

def main():
    print("=== 1. 오프라인 데이터셋(RAM 적재) 테스트 ===")
    # 방금 전 단계에서 만든 npz 파일 경로 지정 (경로가 다르면 수정해주세요)
    offline_dataset = OfflineQECDataset("datasets/d5_complex_noise/val.npz")
    offline_loader = DataLoader(offline_dataset, batch_size=256, shuffle=True)
    
    # 첫 번째 배치 뽑아보기
    x_batch, y_batch = next(iter(offline_loader))
    print(f"입력(X) 형태: {x_batch.shape} -> (Batch, Channel, Detectors)")
    print(f"정답(Y) 형태: {y_batch.shape}")
    
    
    print("\n=== 2. 온라인 데이터셋(실시간 생성) 테스트 ===")
    code_config = CodeParams(distance=5, rounds=5)
    noise_config = NoiseParams(p_gate=0.005, p_meas=0.005, p_corr=0.001, p_leak=0.02)
    
    # 1 에포크당 1만 개만 실시간으로 생성하도록 설정
    online_dataset = OnlineQECDataset(code_config, noise_config, epoch_size=10000)
    # 온라인 데이터셋은 Iterable이므로 shuffle 옵션을 주지 않습니다.
    online_loader = DataLoader(online_dataset, batch_size=256)
    
    start_time = time.time()
    batch_count = 0
    for x_batch, y_batch in online_loader:
        batch_count += 1
    
    elapsed = time.time() - start_time
    print(f"10,000개 실시간 생성 및 로드 완료! (생성된 배치 수: {batch_count})")
    print(f"소요 시간: {elapsed:.2f}초")

if __name__ == "__main__":
    main()