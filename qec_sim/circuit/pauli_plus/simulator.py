# qec_sim/circuit/pauli_plus/simulator.py
"""
PauliPlusSimulator — Pauli frame 기반 시뮬레이터 (Bausch et al. 2023).

Stim은 회로 구조(qubit 좌표, CX 레이어, observable 정의) 추출에만 사용.
실제 에러 추적은 PauliFrame이 담당.

Phase 1: SI1000 depolarizing noise
Phase 2: Leakage (L2/L3) + IQ soft measurement
Phase 3: Crosstalk (ZZ-dominant phenomenological)

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
from .frame import PauliFrame, _CX_C, _CX_T, apply_h_batch_nb, apply_cx_layer_nb, measure_z_batch_nb
from .channels import (depolarize1, depolarize2, bitflip,
                       depolarize1_batch_nb, depolarize2_layer_nb, bitflip_batch_nb,
                       PAULI_MUL, _PAULI_PAIRS)
from .leakage import (remove_leakage, apply_passive_decay,
                      _apply_cz_leakage_layer_nb, _apply_heating_nb)
from .iq_noise import (_get_true_z_state_batch_nb, _sample_iq_batch_nb,
                       _compute_posterior_batch_nb)
from .crosstalk import apply_crosstalk, build_crosstalk_arrays


class PauliPlusSimulator(BaseSimulator):

    def __init__(self, code_params: CodeParams, noise_params: PauliPlusNoiseParams,
                 seed: int | None = None,
                 reference_circuit=None):
        self.code = code_params
        self.noise = noise_params
        self.rng = np.random.default_rng(seed)

        # 참조 회로: code.name으로 등록된 빌더의 출력을 그대로 사용 (m2d_converter는
        # noise instruction을 무시하므로 noise 0 또는 임의 noise 모두 OK).
        if reference_circuit is None:
            from qec_sim.config.schema import NoiseParams
            from qec_sim.circuit.registry import build_circuit
            zero = NoiseParams(p_gate=0.0, p_meas=0.0, p_corr=0.0)
            reference_circuit = build_circuit(code_params.name, code_params, zero).build()

        self._layout = _SurfaceCodeLayout(
            code_params.distance, code_params.rounds, circuit=reference_circuit
        )
        self._nd = self._layout.num_detectors
        self._no = self._layout.num_observables

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

    def reseed(self, seed) -> None:
        """np.random.SeedSequence | int | None → 자체 rng 재생성."""
        self.rng = np.random.default_rng(seed)

    def get_ancilla_coordinates(self) -> dict:
        """ancilla_order 기준 인덱스 → (x, y) 좌표 반환. SoftGridPreprocessor 초기화에 사용."""
        coords = self._layout.qubit_coords
        return {i: (coords[q][0], coords[q][1])
                for i, q in enumerate(self._layout.ancilla_order)}

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
        _reset(frame, layout.all_qubits_arr, noise.p_reset, rng)

        # 측정 기록
        ancilla_meas     = []                    # hard: (shots, n_ancilla) bool
        soft_meas_rounds = []                    # soft: (shots, n_ancilla) float, IQ on시만
        use_iq           = noise.snr > 0
        post1_meas       = None

        # 배치 연산에 쓸 상수 참조
        pm        = PAULI_MUL
        pp        = _PAULI_PAIRS
        cx_c      = _CX_C
        cx_t      = _CX_T
        x_anc     = layout.x_ancilla_arr
        anc_order = layout.ancilla_order_arr
        data_arr  = layout.data_arr
        n_x       = len(x_anc)
        n_anc     = layout.n_ancilla
        n_data    = layout.n_data
        p_decay   = 1.0 - np.exp(-noise.t_meas_over_T1) if noise.t_meas_over_T1 > 0 else 0.0
        p_heat    = noise.heating_rate_per_us * noise.t_cx_us

        for r in range(rounds):
            # 1. H on X-ancilla (배치) + 1q gate noise (배치)
            apply_h_batch_nb(frame.frame, x_anc)
            if noise.p_1q > 0:
                depolarize1_batch_nb(frame.frame, x_anc, noise.p_1q,
                                     rng.random((shots, n_x)), pm)

            # 2. CX 4 sub-layers
            for li, (ctrls, tgts) in enumerate(layout.cx_layer_arrays):
                n_pairs = len(ctrls)

                # CX Clifford (배치)
                apply_cx_layer_nb(frame.frame, ctrls, tgts, cx_c, cx_t)

                # 2q depolarizing (배치)
                if noise.p_2q > 0:
                    depolarize2_layer_nb(
                        frame.frame, ctrls, tgts, noise.p_2q,
                        rng.integers(0, 15, size=(shots, n_pairs), dtype=np.int32),
                        rng.random((shots, n_pairs)),
                        pp, pm,
                    )

                # Phase 2: leakage + heating (배치)
                if noise.cz_dephasing_leakage > 0:
                    _apply_cz_leakage_layer_nb(
                        frame.frame, ctrls, tgts, noise.cz_dephasing_leakage,
                        rng.random((shots, n_pairs)),   # rand_ctrl
                        rng.random((shots, n_pairs)),   # rand_tgt
                        rng.random((shots, n_pairs)),   # rand_dep_ctrl
                        rng.random((shots, n_pairs)),   # rand_dep_tgt
                        pm,
                    )
                if noise.heating_rate_per_us > 0:
                    active_q = layout.active_per_layer[li]
                    _apply_heating_nb(frame.frame, active_q, p_heat,
                                      rng.random((shots, len(active_q))))

                # idle noise (배치)
                idle = layout.idle_per_layer[li]
                if noise.p_idle > 0 and len(idle) > 0:
                    depolarize1_batch_nb(frame.frame, idle, noise.p_idle,
                                         rng.random((shots, len(idle))), pm)

                # Phase 3: crosstalk (precomputed array batch)
                ct_q0, ct_q1 = layout.crosstalk_pair_arrays[li]
                apply_crosstalk(frame, ct_q0, ct_q1,
                                layout.crosstalk_leakage_qubit_arrays[li],
                                p_crosstalk=noise.p_crosstalk,
                                alpha=noise.crosstalk_alpha,
                                crosstalk_leakage_rate=noise.crosstalk_leakage_rate,
                                rng=rng)

            # 3. H on X-ancilla (배치) + 1q gate noise (배치)
            apply_h_batch_nb(frame.frame, x_anc)
            if noise.p_1q > 0:
                depolarize1_batch_nb(frame.frame, x_anc, noise.p_1q,
                                     rng.random((shots, n_x)), pm)

            # 4. Data qubit resonator idle (배치)
            if noise.p_resonator_idle > 0:
                depolarize1_batch_nb(frame.frame, data_arr, noise.p_resonator_idle,
                                     rng.random((shots, n_data)), pm)
            # Phase 2: passive T1 decay (data qubits 일괄 처리, 측정 중 시간)
            if noise.passive_decay_T1_us > 0:
                apply_passive_decay(frame, layout.data_arr,
                                    noise.passive_decay_T1_us,
                                    t_us=noise.t_meas_us, rng=rng)

            # 5. Measure ancilla in Z + IQ noise
            if use_iq:
                true_states = _get_true_z_state_batch_nb(frame.frame, anc_order)
                z_batch     = _sample_iq_batch_nb(
                    true_states, noise.snr, p_decay,
                    rng.standard_normal((shots, n_anc)),
                    rng.random((shots, n_anc)),
                )
                post1_meas, _ = _compute_posterior_batch_nb(z_batch, noise.snr, p_decay)

            if noise.p_meas > 0:
                bitflip_batch_nb(frame.frame, anc_order, noise.p_meas,
                                 rng.random((shots, n_anc)), pm)
            meas = measure_z_batch_nb(frame.frame, anc_order,
                                      rng.random((shots, n_anc)))

            ancilla_meas.append(meas)
            if use_iq:
                soft_meas_rounds.append(post1_meas)

            # 6. 리셋 ancilla + leakage removal
            _reset(frame, anc_order, noise.p_reset, rng)
            if noise.leakage_removal_after_reset:
                remove_leakage(frame, layout.ancilla_order, rng)

            # Phase 2: data qubit leakage removal per cycle
            if noise.data_qubit_leakage_removal_per_cycle:
                remove_leakage(frame, layout.data_qubits, rng)

        # --- 최종 data qubit 측정 (항상 hard — label 누출 방지) ---
        if noise.p_meas > 0:
            bitflip_batch_nb(frame.frame, layout.data_arr, noise.p_meas,
                             rng.random((shots, n_data)), pm)
        data_meas = measure_z_batch_nb(frame.frame, layout.data_arr,
                                       rng.random((shots, n_data)))

        # --- 측정 기록을 stim 순서로 concat → m2d_converter로 detection events + observables 추출 ---
        # Stim 측정 순서: MR_0, MR_1, ..., MR_{T-1}, M (final data).
        # 우리 ancilla_meas[r]는 layout.ancilla_order (= 첫 MR 순서)로 기록되고,
        # data_meas는 layout.data_order (= 최종 M 순서)로 기록되므로 그대로 concat 가능.
        all_meas = np.concatenate([*ancilla_meas, data_meas], axis=1).astype(bool)
        syndromes, logical = layout._m2d_converter.convert(
            measurements=all_meas, separate_observables=True
        )

        # soft_measurements: (shots, rounds, n_ancilla) float — IQ on일 때만
        if use_iq and soft_meas_rounds:
            soft_measurements = np.stack(soft_meas_rounds, axis=1)
        else:
            soft_measurements = None

        return {
            'syndromes':         syndromes,
            'observables':       logical,
            'soft_measurements': soft_measurements,
            'leakage_flags':     None,   # Phase 3
        }


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _reset(frame: PauliFrame, qubits, p: float, rng):
    """Reset to |0⟩ (frame 클리어) + bitflip 노이즈. qubits는 np.int32 배열 권장."""
    q_arr = qubits if isinstance(qubits, np.ndarray) else np.array(qubits, dtype=np.int32)
    frame.frame[:, q_arr] = 0
    if p > 0:
        bitflip_batch_nb(frame.frame, q_arr, p,
                         rng.random((frame.n_shots, len(q_arr))), PAULI_MUL)


# ------------------------------------------------------------------ #
# Layout                                                              #
# ------------------------------------------------------------------ #

class _SurfaceCodeLayout:
    """
    Stim에서 회로 구조(qubit 역할, CX 레이어, observable)를 추출.
    노이즈 없는 회로만 사용 — Pauli frame이 에러를 직접 추적.
    """

    def __init__(self, d: int, rounds: int, *, circuit):
        if circuit is None:
            raise ValueError(
                "_SurfaceCodeLayout requires a stim.Circuit (PauliPlusSimulator는 "
                "code.name으로 등록된 빌더에서 자동 생성. 빌더가 없으면 등록 필요)."
            )
        self.d = d
        self.rounds = rounds
        circ = circuit
        self._reference_circuit = circ
        # m2d_converter는 raw 측정 → (detection events, observables)를 Stim의 정확한
        # detector 정의에 맞게 변환. 우리가 round 0 / round R / final 분기 logic을
        # 직접 만들 필요 없음 — 어떤 stim 회로(memory_x, color_code 등)든 자동.
        self._m2d_converter = circ.compile_m2d_converter()
        self.n_qubits = circ.num_qubits
        self.num_observables = circ.num_observables
        coords = circ.get_final_qubit_coordinates()
        self.qubit_coords = coords  # get_ancilla_coordinates() 재사용

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

        # ancilla_order = Stim MR 순서 (측정 기록을 stim 순서로 concat할 때 사용)
        self.ancilla_order = self._get_mr_order(circ)

        # data_order = Stim M (최종 data 측정) 순서
        self.data_order = self._get_m_order(circ)

        # all_qubits = frame에서 사용하는 모든 qubit 인덱스
        self.all_qubits = list(coords.keys())
        self.all_qubits_arr = np.array(self.all_qubits, dtype=np.int32)

        # CX layers: 4 sub-layers, 순서 보존
        self.cx_layers = self._get_cx_layers(circ)

        # logical Z 정보는 m2d_converter가 자체 처리하므로 별도 추출 불필요.
        # detector 수도 stim에서 직접.
        self.num_detectors = circ.num_detectors

        # --- 배치 연산용 numpy 배열 (미리 계산) ---
        self.x_ancilla_arr    = np.array(x_anc,                dtype=np.int32)
        self.ancilla_order_arr = np.array(self.ancilla_order,  dtype=np.int32)
        self.data_arr         = np.array(data,                 dtype=np.int32)

        # CX sub-layer별 (ctrls, tgts) numpy 배열
        self.cx_layer_arrays = [
            (np.array([p[0] for p in layer], dtype=np.int32),
             np.array([p[1] for p in layer], dtype=np.int32))
            for layer in self.cx_layers
        ]

        # sub-layer별 idle qubit 배열 + active qubit 배열 (heating용)
        self.idle_per_layer   = []
        self.active_per_layer = []
        for layer in self.cx_layers:
            active = set(q for pair in layer for q in pair)
            idle   = [q for q in self.all_qubits if q not in active]
            self.idle_per_layer.append(np.array(idle, dtype=np.int32))
            self.active_per_layer.append(np.array(sorted(active), dtype=np.int32))

        # sub-layer별 crosstalk pair 배열 + leakage 큐빗 배열 (한 번만 계산)
        self.crosstalk_pair_arrays = []           # list of (q0_arr, q1_arr)
        self.crosstalk_leakage_qubit_arrays = []  # list of qubit_arr
        for layer in self.cx_layers:
            q0, q1, leak_q = build_crosstalk_arrays(layer)
            self.crosstalk_pair_arrays.append((q0, q1))
            self.crosstalk_leakage_qubit_arrays.append(leak_q)

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

