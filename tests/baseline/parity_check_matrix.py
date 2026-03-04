import stim
import pymatching

# 1. Surface Code 회로 생성 (Distance 3, 1라운드 예시)
# 사용자님의 builder.py에서 사용하는 방식과 유사합니다.
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=1,
    after_clifford_depolarization=0.01
)

# 2. 회로에서 Detector Error Model (DEM) 추출
# 이 DEM이 사실상 '에러(열) -> 신드롬(행)'을 매핑하는 패리티 검사 행렬 역할을 합니다.
error_model = circuit.detector_error_model(decompose_errors=True)

print(f"총 에러 메커니즘 수 (행렬의 열): {error_model.num_errors}")
print(f"총 디텍터 수 (행렬의 행): {error_model.num_detectors}")

# 3. 추출된 DEM을 바탕으로 PyMatching의 매칭 그래프(Tanner Graph) 생성
matching = pymatching.Matching.from_detector_error_model(error_model)

# 4. (참고) 매칭 그래프의 구조를 NetworkX 그래프로 변환해 내부 구조를 볼 수도 있습니다.
nx_graph = matching.to_networkx()
print(f"그래프 노드(디텍터) 수: {nx_graph.number_of_nodes()}")
print(f"그래프 엣지(에러) 수: {nx_graph.number_of_edges()}")

import stim
import numpy as np

# 1. 간단한 3큐비트 회로 준비 (상태: |000> + |111>)
# H 연산자로 0번 큐비트를 중첩시키고, CX로 1, 2번 큐비트를 얽힘
encoding_circuit = stim.Circuit("""
    H 0
    CX 0 1
    CX 0 2
""")

# 2. 회로 상태를 나타내는 Tableau 추출
tableau = stim.Tableau.from_circuit(encoding_circuit)

# 3. 안정자(Stabilizers) 생성자 추출
stabilizers = tableau.to_stabilizers()

print("==== 추출된 Stabilizers ====")
for stab in stabilizers:
    print(stab)
# 출력 예상: 
# +X0*X1*X2
# +Z0*Z1
# +Z1*Z2

# 4. Stabilizer를 Hx, Hz Numpy 행렬로 변환
Hx_list = []
Hz_list = []

for stab in stabilizers:
    # xs, zs는 각 큐비트 위치의 X, Z 여부를 boolean 배열로 반환
    xs, zs = stim.PauliString(stab).to_numpy()
    
    Hx_list.append(xs.astype(int)) # True/False -> 1/0 변환
    Hz_list.append(zs.astype(int))

Hx = np.array(Hx_list)
Hz = np.array(Hz_list)

print("\n==== H_X (X 에러 체크 행렬) ====")
print(Hx)
# 출력 예상:
# [[1 1 1]
#  [0 0 0]
#  [0 0 0]]

print("\n==== H_Z (Z 에러 체크 행렬) ====")
print(Hz)
# 출력 예상:
# [[0 0 0]
#  [1 1 0]
#  [0 1 1]]