# tests/baseline/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os

from qec_sim.data.dataset import OfflineQECDataset, OnlineQECDataset
from qec_sim.core.parameters import CodeParams, NoiseParams

# 우리가 만든 레지스트리에서 모델 불러오기 함수 임포트
from qec_sim.models import build_model

def main(config_path):
    # 1. YAML 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"[{config_path}] 설정으로 학습을 시작합니다...")

    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # 2. 데이터셋 및 데이터로더 설정
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 512)
    epochs = train_config.get('epochs', 20)
    lr = train_config.get('learning_rate', 0.001)
    data_mode = train_config.get('data_mode', 'offline')
    
    print(f"\n데이터 로드 모드: {data_mode}")
    if data_mode == 'offline':
        train_path = train_config.get('train_path', 'datasets/d5_complex_noise/train.npz')
        val_path = train_config.get('val_path', 'datasets/d5_complex_noise/val.npz')
        train_dataset = OfflineQECDataset(train_path)
        val_dataset = OfflineQECDataset(val_path)
        
        # 데이터 형태 파악
        sample_x, sample_y = train_dataset[0]
        num_detectors = sample_x.shape[1]
        num_observables = sample_y.shape[0]
        
    elif data_mode == 'online':
        code_config = CodeParams(**config['code'])
        noise_config = NoiseParams(**config['noise'])
        
        # IterableDataset 생성
        train_dataset = OnlineQECDataset(code_config, noise_config, epoch_size=train_config.get('epoch_size', 100000))
        val_dataset = OnlineQECDataset(code_config, noise_config, epoch_size=train_config.get('val_size', 10000))
        
        # 임시 회로를 만들어 디텍터/옵저버블 수 추출
        from qec_sim.core.builder import CustomCircuitBuilder
        temp_circuit = CustomCircuitBuilder(code_config, noise_config).build()
        num_detectors = temp_circuit.num_detectors
        num_observables = temp_circuit.num_observables
    else:
        raise ValueError("data_mode는 'offline' 또는 'online'이어야 합니다.")

    # [주의] OnlineDataset은 Iterable이므로 shuffle=False로 둬야 합니다.
    is_iterable = data_mode == 'online'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not is_iterable)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"디텍터 수: {num_detectors}, 논리 큐비트 수: {num_observables}")

    # 3. 모델 동적 생성 (레지스트리 이용)
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'erasure_mlp')
    model_kwargs = model_config.get('kwargs', {})
    
    # 설정 파일에 적힌 이름과 인자로 모델을 조립합니다.
    model = build_model(model_name, num_detectors=num_detectors, num_observables=num_observables, **model_kwargs).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 학습 루프
    print("\n🚀 본격적인 학습을 시작합니다!")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            steps += batch_x.size(0)
            
        train_loss /= steps

        # 5. 검증 루프
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # 예측값 도출 (0.5 이상이면 1, 아니면 0)
                predictions = (outputs > 0).float()
                correct_predictions += (predictions == batch_y).all(dim=1).sum().item()
                val_steps += batch_x.size(0)
                
        val_loss /= val_steps
        logical_error_rate = 1.0 - (correct_predictions / val_steps)

        print(f"[Epoch {epoch+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Logical Error Rate: {logical_error_rate * 100:.2f}%")

    # 6. 모델 가중치 저장
    save_path = train_config.get('save_path', 'model_weights.pth')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True) # 폴더가 없으면 생성
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 학습 완료! 모델 가중치가 '{save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_mlp.yaml", help="YAML 설정 파일 경로")
    args = parser.parse_args()
    main(args.config)