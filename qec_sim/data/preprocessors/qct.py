# qec_sim/data/preprocessors/qct.py
import torch
from collections import defaultdict
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor


def _extract_qubit_coords(circuit) -> Dict[int, tuple]:
    """Stim 회로에서 QUBIT_COORDS instruction을 모아 {qubit_idx: (x, y, ...)} 반환."""
    out = {}
    for instr in circuit.flattened():
        if instr.name != "QUBIT_COORDS":
            continue
        coords = tuple(instr.gate_args_copy())
        for tgt in instr.targets_copy():
            out[tgt.qubit_value] = coords
    return out


def _classify_rotated_surface_qubits(qubit_coords):
    """Rotated surface code 좌표 규칙에 따라 data/Z-anc/X-anc 분류.
       data: (x, y) 모두 odd. Z-anc: (x-y) % 4 == 0. X-anc: (x-y) % 4 == 2.
    """
    data, z_anc, x_anc = {}, {}, {}
    for q, (x, y) in qubit_coords.items():
        ix, iy = int(x), int(y)
        if ix % 2 == 1 and iy % 2 == 1:
            data[q] = (ix, iy)
        elif (ix - iy) % 4 == 0:
            z_anc[q] = (ix, iy)
        elif (ix - iy) % 4 == 2:
            x_anc[q] = (ix, iy)
        else:
            raise RuntimeError(f"unexpected qubit {q} at ({x},{y})")
    return data, z_anc, x_anc


@register_preprocessor("qct_qubit_centric")
class QCTPreprocessor(BasePreprocessor):
    """QCT (Qubit-centric Transformer)용 전처리기.

    Stim 회로의 detector_coords + qubit_coords로부터 다음을 계산해 모델에 주입:
      - n: 데이터 큐빗 수
      - m_Z, m_X: Z/X-stab ancilla 수
      - T_Z, T_X: Z/X-stab이 detector를 갖는 round 수
      - K_Z, K_X: 큐빗당 인접 Z/X-stab 최대 개수 (rotated SC에선 ≤2)
      - nbr_Z (n, K_Z): 데이터 큐빗 i의 인접 Z-stab ancilla idx (-1 = 패딩)
      - nbr_X (n, K_X)
      - det_Z_at (m_Z, T_Z): Z-stab a, round t의 detector idx
      - det_X_at (m_X, T_X)
      - mask (n, n): N(i) ∩ N(j) ≠ ∅ → True (unmasked)

    `gpu_transform`은 syndromes (B, num_detectors)를 그대로 float로 반환 — gather와
    sign 변환은 모델 forward에서 수행 (학습 가능한 W_e와 결합되므로).
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        self.num_detectors = circuit.num_detectors
        self._required_keys = ["syndromes"]
        self._build_topology(circuit)

    def _build_topology(self, circuit):
        qc = _extract_qubit_coords(circuit)
        data_q, z_anc_q, x_anc_q = _classify_rotated_surface_qubits(qc)

        # row-major (y, x) 정렬 — 결정론적 인덱싱
        data_sorted = sorted(data_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))
        z_sorted = sorted(z_anc_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))
        x_sorted = sorted(x_anc_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))

        n = len(data_sorted)
        m_Z = len(z_sorted)
        m_X = len(x_sorted)

        z_coord_to_idx = {coord: i for i, (_, coord) in enumerate(z_sorted)}
        x_coord_to_idx = {coord: i for i, (_, coord) in enumerate(x_sorted)}

        nbr_Z, nbr_X = [[] for _ in range(n)], [[] for _ in range(n)]
        for di, (_, (x, y)) in enumerate(data_sorted):
            for dx, dy in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
                c = (x + dx, y + dy)
                if c in z_coord_to_idx:
                    nbr_Z[di].append(z_coord_to_idx[c])
                elif c in x_coord_to_idx:
                    nbr_X[di].append(x_coord_to_idx[c])

        K_Z = max(len(v) for v in nbr_Z)
        K_X = max(len(v) for v in nbr_X)
        nbr_Z_t = torch.full((n, K_Z), -1, dtype=torch.long)
        nbr_X_t = torch.full((n, K_X), -1, dtype=torch.long)
        for i in range(n):
            if nbr_Z[i]:
                nbr_Z_t[i, :len(nbr_Z[i])] = torch.tensor(nbr_Z[i], dtype=torch.long)
            if nbr_X[i]:
                nbr_X_t[i, :len(nbr_X[i])] = torch.tensor(nbr_X[i], dtype=torch.long)

        det_coords = circuit.get_detector_coordinates()
        z_det_per_round = defaultdict(dict)
        x_det_per_round = defaultdict(dict)
        for det_id, (cx, cy, ct) in det_coords.items():
            coord = (int(cx), int(cy))
            if coord in z_coord_to_idx:
                z_det_per_round[ct][z_coord_to_idx[coord]] = det_id
            elif coord in x_coord_to_idx:
                x_det_per_round[ct][x_coord_to_idx[coord]] = det_id
            else:
                raise RuntimeError(f"detector {det_id} at ({cx},{cy}) matches no ancilla")

        z_rounds = sorted(z_det_per_round.keys())
        x_rounds = sorted(x_det_per_round.keys())
        T_Z = len(z_rounds)
        T_X = len(x_rounds)

        det_Z_at = torch.zeros((m_Z, T_Z), dtype=torch.long)
        det_X_at = torch.zeros((m_X, T_X), dtype=torch.long)
        for ti, t in enumerate(z_rounds):
            for ai in range(m_Z):
                det_Z_at[ai, ti] = z_det_per_round[t][ai]
        for ti, t in enumerate(x_rounds):
            for ai in range(m_X):
                det_X_at[ai, ti] = x_det_per_round[t][ai]

        # Structure-aware mask (rotated SC: 코너 4, 엣지 6, 인테리어 9)
        mask = torch.zeros((n, n), dtype=torch.bool)
        nbrs_per_qubit = []
        for i in range(n):
            s = set()
            for a in nbr_Z[i]:
                s.add(("Z", a))
            for a in nbr_X[i]:
                s.add(("X", a))
            nbrs_per_qubit.append(s)
        for i in range(n):
            for j in range(n):
                if nbrs_per_qubit[i] & nbrs_per_qubit[j]:
                    mask[i, j] = True

        # 버퍼 등록 — wrapper.to(device) 시 자동 이동
        self.register_buffer("nbr_Z", nbr_Z_t, persistent=False)
        self.register_buffer("nbr_X", nbr_X_t, persistent=False)
        self.register_buffer("det_Z_at", det_Z_at, persistent=False)
        self.register_buffer("det_X_at", det_X_at, persistent=False)
        self.register_buffer("attn_mask", mask, persistent=False)

        self._meta = dict(n=n, m_Z=m_Z, m_X=m_X, T_Z=T_Z, T_X=T_X, K_Z=K_Z, K_X=K_X)

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        # 토폴로지 메타 + 버퍼 텐서를 모델 __init__에 전달
        return dict(
            **self._meta,
            nbr_Z=self.nbr_Z,
            nbr_X=self.nbr_X,
            det_Z_at=self.det_Z_at,
            det_X_at=self.det_X_at,
            attn_mask=self.attn_mask,
        )

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch_data["syndromes"].float()

