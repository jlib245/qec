# qec_sim/circuit/pauli_plus/simulator.py
"""
PauliPlusSimulator — Phase 1: SI1000-style depolarizing noise on rotated surface code.

Noise strengths (SI1000):
  - Z/X measurement: bitflip/phaseflip before measure  = 5p
  - Z/X reset: bitflip/phaseflip after reset           = 2p
  - Resonator idle: 1q depolarizing                    = 2p
  - 2q gate (CZ): 2q depolarizing                      = p
  - 1q Clifford / idle: 1q depolarizing                = p/10
"""

import numpy as np
import stim  # used only for layout/coordinate metadata

from qec_sim.core.interfaces import BaseSimulator
from qec_sim.config.schema import CodeParams, NoiseParams
from .frame import PauliFrame
from .channels import depolarize1, depolarize2, bitflip, phaseflip


class PauliPlusSimulator(BaseSimulator):
    """
    Pauli-frame simulator for rotated surface code.
    Compatible with SimulatorPool (implements BaseSimulator interface).
    """

    def __init__(self, code_params: CodeParams, noise_params: NoiseParams,
                 seed: int | None = None):
        self.code = code_params
        self.noise = noise_params
        self.d = code_params.distance
        self.rounds = code_params.rounds
        self.rng = np.random.default_rng(seed)

        p = float(noise_params.p_gate)
        self.p = p
        self.p_meas = float(noise_params.p_meas)

        # Build layout from Stim (coordinate metadata only, no sampling)
        self._layout = _RotatedSurfaceCodeLayout(self.d)
        self._nd = self._layout.num_detectors(self.rounds)
        self._no = 1  # single Z-logical observable

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
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(self, shots: int) -> dict:
        layout = self._layout
        d = self.d
        p = self.p
        p_meas = self.p_meas
        rounds = self.rounds
        rng = self.rng

        n_ancilla = layout.n_ancilla
        n_qubits = layout.n_qubits  # covers all non-contiguous Stim qubit indices

        frame = PauliFrame(shots, n_qubits)

        # measurement records: (shots, rounds+1, n_ancilla) — round 0 is reset baseline
        meas = np.zeros((shots, rounds + 1, n_ancilla), dtype=bool)

        # --- round 0: reset all, measure ancillas (baseline) ---
        _reset_z(frame, list(range(n_qubits)), 2 * p, rng)
        meas[:, 0, :] = _measure_ancillas(frame, layout, p_meas, rng)

        # --- syndrome extraction rounds ---
        for r in range(1, rounds + 1):
            _idle_data(frame, layout, 2 * p, rng)
            _cz_layer(frame, layout, p, rng)
            _idle_data(frame, layout, 2 * p, rng)
            meas[:, r, :] = _measure_ancillas(frame, layout, p_meas, rng)
            _reset_ancillas(frame, layout, 2 * p, rng)

        # --- detection events: XOR consecutive rounds ---
        # shape: (shots, rounds, n_ancilla)
        det_events = meas[:, 1:, :] ^ meas[:, :-1, :]
        syndromes = det_events.reshape(shots, -1)  # (shots, rounds * n_ancilla)

        # --- logical observable: Z-logical = XOR of one row of data qubits ---
        data_row = layout.logical_z_row  # list of data qubit indices in top row
        logical = np.zeros((shots, 1), dtype=bool)
        for qi in data_row:
            f = frame.frame[:, qi]
            logical[:, 0] ^= (f == 1) | (f == 2)  # X or Y → bit flip

        erasures = np.zeros_like(syndromes, dtype=bool)

        return {
            'syndromes': syndromes,
            'observables': logical,
            'erasures': erasures,
        }


# ------------------------------------------------------------------ #
# Circuit helpers                                                     #
# ------------------------------------------------------------------ #

def _reset_z(frame: PauliFrame, qubits: list, p: float, rng):
    """Reset to |0⟩, then apply bitflip noise."""
    from .frame import X as _X
    for q in qubits:
        # Reset collapses Z: set frame to I on that qubit
        frame.frame[:, q] = 0
        bitflip(frame, q, p, rng)


def _reset_ancillas(frame: PauliFrame, layout, p: float, rng):
    _reset_z(frame, layout.ancilla_indices, p, rng)


def _measure_ancillas(frame: PauliFrame, layout, p_meas: float, rng) -> np.ndarray:
    """Measure all ancillas (Z for Z-stabilizers, X for X-stabilizers).
    Returns (shots, n_ancilla) bool outcomes."""
    outcomes = np.zeros((frame.n_shots, layout.n_ancilla), dtype=bool)
    for i, (q, kind) in enumerate(zip(layout.ancilla_indices, layout.ancilla_kinds)):
        if kind == 'Z':
            bitflip(frame, q, p_meas, rng)
            outcomes[:, i] = frame.measure_z(q)
        else:
            phaseflip(frame, q, p_meas, rng)
            outcomes[:, i] = frame.measure_x(q)
    return outcomes


def _idle_data(frame: PauliFrame, layout, p: float, rng):
    for q in layout.data_indices:
        depolarize1(frame, q, p, rng)


def _cz_layer(frame: PauliFrame, layout, p: float, rng):
    """Apply CZ gates between ancillas and their data neighbours."""
    for (qa, qd) in layout.cz_pairs:
        depolarize2(frame, qa, qd, p, rng)


# ------------------------------------------------------------------ #
# Layout                                                              #
# ------------------------------------------------------------------ #

class _RotatedSurfaceCodeLayout:
    """
    Minimal layout for rotated surface code distance d.
    Uses Stim to extract qubit coordinates, then assigns indices.
    """

    def __init__(self, d: int):
        import stim
        # Use a noiseless circuit just for coordinate/metadata extraction
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=1,
            after_clifford_depolarization=0,
            before_measure_flip_probability=0,
        )
        coords = circ.get_final_qubit_coordinates()  # {qubit_index: [x, y]}
        n_total = circ.num_qubits

        # Stim encodes data vs ancilla by coordinate parity
        # data qubits: (x+y) % 2 == 0; ancilla: (x+y) % 2 == 1
        # (This is the standard rotated surface code convention)
        data_idx = []
        ancilla_idx = []
        ancilla_kind = []

        # Rotated surface code: data qubits have both x,y odd; ancilla have at least one even.
        # Z-ancilla: x even, y even (bulk Z-stabilizers) or boundary
        # X-ancilla: distinguish by which side of diagonal
        # Simpler: use circuit instructions to identify ancilla (they get H gates)
        ancilla_set = self._find_ancilla_from_circuit(circ)

        for qi, (x, y) in sorted(coords.items()):
            if qi in ancilla_set:
                # Z-ancilla: y even, x even; X-ancilla: x even, y even on other diagonal
                # Heuristic: Z-stabilizers measure data in Z basis (no H), ancilla at even x
                kind = 'X' if qi in self._find_x_ancilla(circ) else 'Z'
                ancilla_idx.append(qi)
                ancilla_kind.append(kind)
            else:
                data_idx.append(qi)

        self.data_indices = data_idx
        self.ancilla_indices = ancilla_idx
        self.ancilla_kinds = ancilla_kind
        self.n_data = len(data_idx)
        self.n_ancilla = len(ancilla_idx)
        # Frame must cover all qubit indices (they are non-contiguous in Stim)
        self.n_qubits = n_total

        # CZ pairs from the circuit instructions
        self.cz_pairs = self._extract_cz_pairs(circ)

        # Logical Z: one row of data qubits at y=0 (top boundary)
        top_y = min(coords[q][1] for q in data_idx)
        self.logical_z_row = [q for q in data_idx if coords[q][1] == top_y]

        self._det_per_round = self.n_ancilla
        self._coords = coords

    def num_detectors(self, rounds: int) -> int:
        return self._det_per_round * rounds

    def _find_ancilla_from_circuit(self, circ) -> set:
        """Ancilla qubits receive H gates (for X-stabilizers) or are measured/reset."""
        reset_qubits = set()
        for inst in circ.flattened():
            if inst.name in ('R', 'RX', 'RZ', 'MR', 'MRX', 'MRZ', 'M'):
                for t in inst.targets_copy():
                    reset_qubits.add(t.value)
        # Data qubits are never reset in the rotated surface code (only ancilla are)
        # Actually in Stim's generated circuit, ancilla are reset each round
        # Use: ancilla = qubits that appear in RESET instructions every round
        # Simpler: ancilla have even x or even y in rotated surface code
        coords = circ.get_final_qubit_coordinates()
        ancilla = set()
        for qi, (x, y) in coords.items():
            if int(x) % 2 == 0 or int(y) % 2 == 0:
                ancilla.add(qi)
        return ancilla

    def _find_x_ancilla(self, circ) -> set:
        """X-ancilla: qubits that receive H gates before measurement."""
        h_qubits = set()
        for inst in circ.flattened():
            if inst.name == 'H':
                for t in inst.targets_copy():
                    h_qubits.add(t.value)
        return h_qubits

    def _extract_cz_pairs(self, circ: 'stim.Circuit') -> list:
        pairs = []
        seen = set()
        for inst in circ.flattened():
            if inst.name in ('CZ', 'CNOT', 'CX'):
                targets = inst.targets_copy()
                for i in range(0, len(targets), 2):
                    a, b = targets[i].value, targets[i + 1].value
                    key = (min(a, b), max(a, b))
                    if key not in seen:
                        pairs.append((a, b))
                        seen.add(key)
        return pairs
