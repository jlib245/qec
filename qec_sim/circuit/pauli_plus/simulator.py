# qec_sim/circuit/pauli_plus/simulator.py
"""
PauliPlusSimulator — Pauli frame 기반 시뮬레이터 (Bausch et al. 2023).

Stim은 회로 구조(qubit 좌표, CX 레이어, observable 정의) 추출에만 사용.
실제 에러 추적은 PauliFrame이 담당.

Phase 1: SI1000 depolarizing noise
Phase 2: Leakage (L2/L3) — 미구현
Phase 3: Crosstalk, IQ noise — 미구현

SI1000 noise strengths:
  - 2q gate (CX):          2q depolarizing   p
  - 1q gate (H):           1q depolarizing   p/10
  - Measurement:           bitflip before    5p
  - Reset (after):         bitflip           2p
  - Resonator idle (data
    qubits during meas):   1q depolarizing   2p
  - 1q idle:               1q depolarizing   p/10
"""

import numpy as np
import stim

from qec_sim.core.interfaces import BaseSimulator
from qec_sim.config.schema import CodeParams, PauliPlusNoiseParams
from .frame import PauliFrame
from .channels import depolarize1, depolarize2, bitflip


class PauliPlusSimulator(BaseSimulator):

    def __init__(self, code_params: CodeParams, noise_params: PauliPlusNoiseParams,
                 seed: int | None = None):
        self.code = code_params
        self.noise = noise_params
        self.rng = np.random.default_rng(seed)

        self._layout = _SurfaceCodeLayout(code_params.distance, code_params.rounds)
        self._nd = self._layout.num_detectors
        self._no = 1  # Z-memory: single logical observable

    # ------------------------------------------------------------------ #
    # BaseSimulator interface                                              #
    # ------------------------------------------------------------------ #

    @property
    def num_detectors(self) -> int:
        return self._nd

    @property
    def num_observables(self) -> int:
        return self._no

    def generate_data(self, shots: int) -> dict:
        return self._run(shots)

    # ------------------------------------------------------------------ #
    # Simulation                                                           #
    # ------------------------------------------------------------------ #

    def _run(self, shots: int) -> dict:
        layout = self._layout
        noise  = self.noise
        rounds = self.code.rounds
        rng    = self.rng

        frame = PauliFrame(shots, layout.n_qubits)

        # --- 초기화: 모든 큐빗 리셋 + 리셋 노이즈 ---
        _reset(frame, layout.all_qubits, noise.p_reset, rng)

        # 측정 기록: ancilla_meas[r] = (shots, n_ancilla) bool
        ancilla_meas = []

        for r in range(rounds):
            # 1. H on X-ancilla (+ 1q gate noise)
            for q in layout.x_ancilla:
                frame.apply_h(q)
                depolarize1(frame, q, noise.p_1q, rng)

            # 2. CX 4 sub-layers (+ 2q noise per gate, idle on inactive qubits)
            for layer in layout.cx_layers:
                for (ctrl, tgt) in layer:
                    frame.apply_cx(ctrl, tgt)
                    depolarize2(frame, ctrl, tgt, noise.p_2q, rng)
                active = set(q for pair in layer for q in pair)
                for q in layout.all_qubits:
                    if q not in active:
                        depolarize1(frame, q, noise.p_idle, rng)

            # 3. H on X-ancilla (+ 1q gate noise)
            for q in layout.x_ancilla:
                frame.apply_h(q)
                depolarize1(frame, q, noise.p_1q, rng)

            # 4. Data qubit resonator idle (during ancilla measurement)
            for q in layout.data_qubits:
                depolarize1(frame, q, noise.p_resonator_idle, rng)

            # 5. Measure ancilla in Z (X-ancilla는 H 이미 적용됨 → Z basis로 측정)
            meas = np.zeros((shots, layout.n_ancilla), dtype=bool)
            for i, q in enumerate(layout.ancilla_order):
                bitflip(frame, q, noise.p_meas, rng)
                meas[:, i] = frame.measure_z(q)

            ancilla_meas.append(meas)

            # 6. 리셋 ancilla (+ 리셋 노이즈)
            _reset(frame, layout.ancilla_order, noise.p_reset, rng)

        # --- detection events ---
        # Round 1: Z-ancilla only (4 detectors, 첫 라운드는 이전 측정 없음)
        # Round 2+: all ancilla XOR consecutive (8 detectors each)
        # Final: data qubit 측정 기반 추가 detectors (4개)
        det_list = []

        # round 0: Z-ancilla only vs 0
        z_idx = layout.z_ancilla_indices_in_order
        det_list.append(ancilla_meas[0][:, z_idx])  # (shots, 4)

        # rounds 1..T-1: all 8 ancilla XOR
        for r in range(1, rounds):
            det_list.append(ancilla_meas[r] ^ ancilla_meas[r - 1])  # (shots, 8)

        # --- 최종 data qubit 측정 ---
        data_meas = np.zeros((shots, layout.n_data), dtype=bool)
        for i, q in enumerate(layout.data_order):
            bitflip(frame, q, noise.p_meas, rng)
            data_meas[:, i] = frame.measure_z(q)

        # 최종 round data + 마지막 ancilla 조합 detectors
        final_dets = layout.compute_final_detectors(data_meas, ancilla_meas[-1])
        det_list.append(final_dets)  # (shots, 4)

        syndromes = np.concatenate(det_list, axis=1)  # (shots, num_detectors)

        # --- logical observable: data_order에서 logical_z_indices XOR ---
        logical = np.zeros((shots, 1), dtype=bool)
        for i in layout.logical_z_data_indices:
            logical[:, 0] ^= data_meas[:, i]

        erasures = np.zeros_like(syndromes, dtype=bool)

        return {
            'syndromes':        syndromes,
            'observables':      logical,
            'erasures':         erasures,
            'soft_measurements': None,   # Phase 2
            'leakage_flags':     None,   # Phase 2
        }


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _reset(frame: PauliFrame, qubits, p: float, rng):
    """Reset to |0⟩ (clear frame), then apply bitflip noise."""
    for q in qubits:
        frame.frame[:, q] = 0
        bitflip(frame, q, p, rng)


# ------------------------------------------------------------------ #
# Layout                                                              #
# ------------------------------------------------------------------ #

class _SurfaceCodeLayout:
    """
    Stim에서 회로 구조(qubit 역할, CX 레이어, observable)를 추출.
    노이즈 없는 회로만 사용 — Pauli frame이 에러를 직접 추적.
    """

    def __init__(self, d: int, rounds: int):
        self.d = d
        self.rounds = rounds

        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=rounds,
            after_clifford_depolarization=0,
            before_measure_flip_probability=0,
        )
        self.n_qubits = circ.num_qubits
        coords = circ.get_final_qubit_coordinates()

        # --- qubit 분류 ---
        # data: both x,y odd;  ancilla: at least one even
        x_ancilla_set = self._get_x_ancilla(circ)
        data, z_anc, x_anc = [], [], []

        for qi, (x, y) in sorted(coords.items()):
            if int(x) % 2 == 1 and int(y) % 2 == 1:
                data.append(qi)
            elif qi in x_ancilla_set:
                x_anc.append(qi)
            else:
                z_anc.append(qi)

        self.data_qubits  = data
        self.x_ancilla    = x_anc
        self.z_ancilla    = z_anc
        self.n_data       = len(data)
        self.n_ancilla    = len(x_anc) + len(z_anc)

        # ancilla_order = Stim MR 순서 (detection event 정의와 일치해야 함)
        self.ancilla_order = self._get_mr_order(circ)

        # z_ancilla_indices_in_order: ancilla_order에서 Z-ancilla의 인덱스
        z_set = set(z_anc)
        self.z_ancilla_indices_in_order = [
            i for i, q in enumerate(self.ancilla_order) if q in z_set
        ]

        # data_order = Stim M (최종 data 측정) 순서
        self.data_order = self._get_m_order(circ)

        # all_qubits = frame에서 사용하는 모든 qubit 인덱스
        self.all_qubits = list(coords.keys())

        # CX layers: 4 sub-layers, 순서 보존
        self.cx_layers = self._get_cx_layers(circ)

        # logical Z: data_order에서 observable에 포함된 인덱스
        self.logical_z_data_indices = self._get_logical_z_indices(circ, self.data_order)

        # 최종 detector 정의 (data + ancilla 조합)
        self._final_det_defs = self._get_final_detector_defs(circ, self.data_order, self.ancilla_order)

        # num_detectors: round1(4) + (rounds-1)*8 + final(4)
        self.num_detectors = len(self.z_ancilla_indices_in_order) \
                           + (rounds - 1) * self.n_ancilla \
                           + len(self._final_det_defs)

    def compute_final_detectors(self, data_meas: np.ndarray,
                                 last_ancilla_meas: np.ndarray) -> np.ndarray:
        """
        data_meas: (shots, n_data), last_ancilla_meas: (shots, n_ancilla)
        반환: (shots, n_final_dets)
        """
        shots = data_meas.shape[0]
        result = np.zeros((shots, len(self._final_det_defs)), dtype=bool)
        for col, indices in enumerate(self._final_det_defs):
            for src, idx in indices:
                if src == 'data':
                    result[:, col] ^= data_meas[:, idx]
                else:
                    result[:, col] ^= last_ancilla_meas[:, idx]
        return result

    # ------------------------------------------------------------------ #
    # Stim 회로 파싱                                                       #
    # ------------------------------------------------------------------ #

    def _get_x_ancilla(self, circ) -> set:
        """H 게이트를 받는 qubit = X-ancilla."""
        h_qubits = set()
        for inst in circ.flattened():
            if inst.name == 'H':
                for t in inst.targets_copy():
                    h_qubits.add(t.value)
        return h_qubits

    def _get_mr_order(self, circ) -> list:
        """첫 번째 MR 명령의 qubit 순서."""
        for inst in circ.flattened():
            if inst.name == 'MR':
                return [t.value for t in inst.targets_copy()]
        return []

    def _get_m_order(self, circ) -> list:
        """최종 M (data qubit 측정) 명령의 qubit 순서."""
        last_m = None
        for inst in circ.flattened():
            if inst.name == 'M':
                last_m = inst
        if last_m is None:
            return []
        return [t.value for t in last_m.targets_copy()]

    def _get_cx_layers(self, circ) -> list[list[tuple]]:
        """
        1 round에 해당하는 CX sub-layer 4개를 순서대로 추출.
        REPEAT 블록의 첫 iteration에서 추출.
        """
        layers = []
        for inst in circ:
            if inst.name == 'REPEAT':
                body = inst.body_copy()
                current_layer = []
                for sub in body:
                    if sub.name in ('CX', 'CNOT'):
                        targets = sub.targets_copy()
                        pairs = [(targets[i].value, targets[i+1].value)
                                 for i in range(0, len(targets), 2)]
                        current_layer.extend(pairs)
                    elif sub.name == 'TICK' and current_layer:
                        layers.append(current_layer)
                        current_layer = []
                if current_layer:
                    layers.append(current_layer)
                break
        # REPEAT 없으면 flat하게 추출 (fallback)
        if not layers:
            for inst in circ.flattened():
                if inst.name in ('CX', 'CNOT'):
                    targets = inst.targets_copy()
                    layers.append([(targets[i].value, targets[i+1].value)
                                   for i in range(0, len(targets), 2)])
        return layers

    def _get_logical_z_indices(self, circ, data_order: list) -> list:
        """OBSERVABLE_INCLUDE의 rec 오프셋 → data_order 인덱스로 변환."""
        n_data = len(data_order)
        indices = []
        for inst in circ.flattened():
            if inst.name == 'OBSERVABLE_INCLUDE':
                for t in inst.targets_copy():
                    # rec[-k]: data_order의 (n_data + t.value) 번째 (t.value < 0)
                    idx = n_data + t.value  # t.value는 음수
                    if 0 <= idx < n_data:
                        indices.append(idx)
        return indices

    def _get_final_detector_defs(self, circ, data_order: list, ancilla_order: list) -> list:
        """
        최종 라운드 DETECTOR 정의를 파싱.
        각 detector = [(src, idx), ...] 형태의 리스트.
        src: 'data' or 'ancilla'
        """
        n_data    = len(data_order)
        n_ancilla = len(ancilla_order)

        # 마지막 M 명령 이후의 DETECTOR만 추출
        insts = list(circ.flattened())
        last_m_pos = max(i for i, inst in enumerate(insts) if inst.name == 'M')

        defs = []
        for inst in insts[last_m_pos + 1:]:
            if inst.name != 'DETECTOR':
                continue
            det = []
            for t in inst.targets_copy():
                offset = t.value  # 음수 (rec 오프셋)
                # M은 n_data 개, 그 이전 MR은 n_ancilla 개
                # rec[-1] ~ rec[-n_data]: data
                # rec[-n_data-1] ~ rec[-n_data-n_ancilla]: last ancilla
                if offset >= -n_data:
                    idx = n_data + offset
                    det.append(('data', idx))
                else:
                    idx = n_data + n_ancilla + offset
                    if 0 <= idx < n_ancilla:
                        det.append(('ancilla', idx))
            defs.append(det)
        return defs
