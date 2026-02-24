import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from qec_sim.data.dataset import OfflineQECDataset
from qec_sim.models.baseline import ErasureAwareMLP

def train_model():
    # 1. 하드웨어 설정 (GPU가 있으면 사용, 맥북이면 MPS, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # 2. 데이터 로드
    print("\n데이터를 불러오는 중...")
    train_dataset = OfflineQECDataset("datasets/d5_complex_noise/train.npz")
    val_dataset = OfflineQECDataset("datasets/d5_complex_noise/val.npz")

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # 데이터의 형태를 확인하여 모델 입력 크기 자동 설정
    sample_x, sample_y = train_dataset[0]
    num_detectors = sample_x.shape[1]    # x의 형태: (2채널, num_detectors)
    num_observables = sample_y.shape[0]  # y의 형태: (num_observables,)

    print(f"디텍터 수: {num_detectors}, 논리 큐비트 수: {num_observables}")

    # 3. 모델, 손실 함수, 옵티마이저 초기화
    model = ErasureAwareMLP(num_detectors, num_observables).to(device)
    
    # BCEWithLogitsLoss: 모델의 출력(Logits)에 Sigmoid를 씌우고 이진 교차 엔트로피 계산
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 학습 루프 (Train Loop)
    epochs = 20
    print("\n🚀 본격적인 학습을 시작합니다!")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()           # 기울기 초기화
            outputs = model(batch_x)        # 순전파 (Forward)
            
            loss = criterion(outputs, batch_y) # 손실 계산
            loss.backward()                 # 역전파 (Backward)
            optimizer.step()                # 가중치 업데이트
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_dataset)

        # 5. 검증 루프 (Validation Loop)
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # 예측값 도출 (0.5 이상이면 1, 아니면 0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                # 배치 내에서 논리적 에러가 나지 않은(완벽히 맞춘) 개수 계산
                # (모든 observable을 맞췄을 때 정답으로 인정)
                correct_predictions += (predictions == batch_y).all(dim=1).sum().item()
                
        val_loss /= len(val_dataset)
        logical_error_rate = 1.0 - (correct_predictions / len(val_dataset))

        # 결과 출력
        print(f"[Epoch {epoch+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Logical Error Rate: {logical_error_rate * 100:.2f}%")

if __name__ == "__main__":
    train_model()