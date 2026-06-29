# qec_sim/circuit/pauli_plus/frame.py
"""
Vectorized Pauli frame tracker.

States per qubit (integer encoding):
  I=0, X=1, Y=2, Z=3, L2=4, L3=5
"""

import numpy as np
import numba

# Pauli multiplication table: PAULI_MUL[a, b] = a * b  (ignoring phase)
# I=0, X=1, Y=2, Z=3
_PM = np.zeros((4, 4), dtype=np.int8)
for _i in range(4):
    _PM[_i, 0] = _i
    _PM[0, _i] = _i
_PM[1, 1] = 0  # XX=I
_PM[2, 2] = 0  # YY=I
_PM[3, 3] = 0  # ZZ=I
_PM[1, 2] = 3  # XY=Z
_PM[2, 1] = 3  # YX=Z
_PM[1, 3] = 2  # XZ=Y
_PM[3, 1] = 2  # ZX=Y
_PM[2, 3] = 1  # YZ=X
_PM[3, 2] = 1  # ZY=X
PAULI_MUL = _PM

I, X, Y, Z, L2, L3 = 0, 1, 2, 3, 4, 5

# CX conjugation table: CX_C[p_control, p_target] = new_control,
# CX_T[p_control, p_target] = new_target (Heisenberg: X_c→X_cX_t, Z_t→Z_cZ_t).
# The flat list is laid out as [ft, fc] (target outer, control inner), so we
# transpose after reshape so callers can index as [fc, ft] = [control, target].
_CX_RAW = [
    # ft=I row, varying fc=I,X,Y,Z
    (I,I),(X,X),(Y,X),(Z,I),
    # ft=X row
    (I,X),(X,I),(Y,I),(Z,X),
    # ft=Y row
    (Z,Y),(Y,Z),(X,Z),(I,Y),
    # ft=Z row
    (Z,Z),(Y,Y),(X,Y),(I,Z),
]
_CX_C = np.ascontiguousarray(
    np.array([v[0] for v in _CX_RAW], dtype=np.int8).reshape(4, 4).T
)
_CX_T = np.ascontiguousarray(
    np.array([v[1] for v in _CX_RAW], dtype=np.int8).reshape(4, 4).T
)

class _CXTable:
    """Allows CX_TABLE[fc, ft] → (new_c, new_t) via numpy fancy indexing."""
    def __getitem__(self, key):
        fc, ft = key
        return _CX_C[fc, ft], _CX_T[fc, ft]

CX_TABLE = _CXTable()


class PauliFrame:
    """
    Shape: (n_shots, n_qubits), dtype int8.
    Values 0-3: standard Pauli; 4-5: leakage states (Phase 2).
    """

    def __init__(self, n_shots: int, n_qubits: int):
        self.n_shots = n_shots
        self.n_qubits = n_qubits
        # Start in all-I state
        self.frame = np.zeros((n_shots, n_qubits), dtype=np.int8)

    def apply_single_pauli(self, qubit: int, pauli: np.ndarray):
        """
        pauli: (n_shots,) int8 array of {I,X,Y,Z}.
        Multiply frame[:, qubit] by pauli element-wise.
        """
        f = self.frame[:, qubit]
        # Only update non-leaked qubits
        not_leaked = f < 4
        f[not_leaked] = PAULI_MUL[f[not_leaked], pauli[not_leaked]]
        self.frame[:, qubit] = f

    def apply_two_qubit_pauli(self, q0: int, q1: int,
                               p0: np.ndarray, p1: np.ndarray):
        """p0, p1: (n_shots,) int8 arrays."""
        self.apply_single_pauli(q0, p0)
        self.apply_single_pauli(q1, p1)

    def measure_z(self, qubit: int, rng=None) -> np.ndarray:
        """
        Z-basis measurement. Returns (n_shots,) bool outcome.
          - Normal qubit: outcome = 1 iff X or Y in frame. Collapses X component.
          - Leaked qubit (L2/L3): outcome is random (50/50). Frame set to I after.
        """
        f = self.frame[:, qubit]
        leaked = f >= 4

        outcome = (f == X) | (f == Y)

        if leaked.any():
            _rng = rng if rng is not None else np.random.default_rng()
            outcome[leaked] = _rng.integers(0, 2, size=leaked.sum(), dtype=bool)

        # Collapse: remove X component; leaked → reset to I
        new_f = np.where(f == X, I, np.where(f == Y, Z, f))
        new_f[leaked] = I
        self.frame[:, qubit] = new_f
        return outcome

    def measure_x(self, qubit: int, rng=None) -> np.ndarray:
        """X-basis measurement. Outcome is 1 iff Z or Y. Leaked → random."""
        f = self.frame[:, qubit]
        leaked = f >= 4

        outcome = (f == Z) | (f == Y)

        if leaked.any():
            _rng = rng if rng is not None else np.random.default_rng()
            outcome[leaked] = _rng.integers(0, 2, size=leaked.sum(), dtype=bool)

        new_f = np.where(f == Z, I, np.where(f == Y, X, f))
        new_f[leaked] = I
        self.frame[:, qubit] = new_f
        return outcome

    def apply_h(self, qubit: int):
        """H conjugation: X↔Z, Y→Y. Leaked qubits are unaffected."""
        f = self.frame[:, qubit]
        not_leaked = f < 4
        new_f = f.copy()
        new_f[not_leaked] = np.where(f[not_leaked] == X, Z,
                            np.where(f[not_leaked] == Z, X, f[not_leaked]))
        self.frame[:, qubit] = new_f

    def apply_cx(self, control: int, target: int):
        """
        CX (CNOT) Clifford conjugation on the Pauli frame.
        Rules (Heisenberg): X_c→X_cX_t, Z_t→Z_cZ_t.

        Leakage handling (Pauli+ spec):
          - both leaked: no-op
          - control leaked, target normal: apply 1q depolarizing to target (handled in channels.py)
          - control normal, target leaked: same
          - both normal: standard CX conjugation
        Here we only do the Clifford part; caller applies 1q noise for leaked case.
        """
        fc = self.frame[:, control].copy()
        ft = self.frame[:, target].copy()

        # Shots where both qubits are in computational subspace
        both_normal = (fc < 4) & (ft < 4)

        nc = fc.copy()
        nt = ft.copy()
        nc[both_normal] = _CX_C[fc[both_normal], ft[both_normal]]
        nt[both_normal] = _CX_T[fc[both_normal], ft[both_normal]]

        self.frame[:, control] = nc
        self.frame[:, target] = nt

    def copy(self) -> 'PauliFrame':
        pf = PauliFrame(self.n_shots, self.n_qubits)
        pf.frame = self.frame.copy()
        return pf


# ------------------------------------------------------------------ #
# Numba-JIT batch operations (module-level, cache=True)              #
# ------------------------------------------------------------------ #

@numba.njit(cache=True)
def apply_h_batch_nb(frame, qubits):
    """H 게이트를 여러 qubit에 일괄 적용. frame: (shots, n_qubits) int8"""
    n_shots = frame.shape[0]
    for qi in range(len(qubits)):
        q = qubits[qi]
        for i in range(n_shots):
            f = frame[i, q]
            if f == 1:        # X → Z
                frame[i, q] = 3
            elif f == 3:      # Z → X
                frame[i, q] = 1
            # Y→Y, leaked→leaked: 변화 없음


@numba.njit(cache=True)
def apply_cx_layer_nb(frame, ctrls, tgts, cx_c, cx_t):
    """한 CX sub-layer의 모든 게이트를 일괄 적용."""
    n_shots = frame.shape[0]
    for pi in range(len(ctrls)):
        c = ctrls[pi]
        t = tgts[pi]
        for i in range(n_shots):
            fc = frame[i, c]
            ft = frame[i, t]
            if fc < 4 and ft < 4:
                frame[i, c] = cx_c[fc, ft]
                frame[i, t] = cx_t[fc, ft]


@numba.njit(cache=True)
def measure_z_batch_nb(frame, qubits, leak_rand):
    """
    여러 qubit을 Z-basis로 일괄 측정.
    leak_rand: (shots, n_qubits) float — leaked qubit 무작위 결과용.
    반환: (shots, n_qubits) bool
    """
    n_shots = frame.shape[0]
    n_q = len(qubits)
    outcomes = np.zeros((n_shots, n_q), dtype=np.bool_)
    for qi in range(n_q):
        q = qubits[qi]
        for i in range(n_shots):
            f = frame[i, q]
            if f >= 4:                       # leaked → 50/50
                outcomes[i, qi] = leak_rand[i, qi] < 0.5
                frame[i, q] = 0              # I로 리셋
            else:
                outcomes[i, qi] = (f == 1) or (f == 2)   # X or Y
                if f == 1:                   # X → I
                    frame[i, q] = 0
                elif f == 2:                 # Y → Z
                    frame[i, q] = 3
    return outcomes
