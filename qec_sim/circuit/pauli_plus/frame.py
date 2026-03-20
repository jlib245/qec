# qec_sim/circuit/pauli_plus/frame.py
"""
Vectorized Pauli frame tracker.

States per qubit (integer encoding):
  I=0, X=1, Y=2, Z=3, L2=4, L3=5
"""

import numpy as np

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

    def measure_z(self, qubit: int) -> np.ndarray:
        """
        Returns (n_shots,) bool: measurement outcome (True = flipped).
        Z-basis: outcome is 1 iff frame has X or Y component.
        After measurement, Z component is reset to I on that qubit.
        """
        f = self.frame[:, qubit]
        outcome = (f == X) | (f == Y)
        # Collapse: remove X component, keep Z
        new_f = np.where(f == X, I, np.where(f == Y, Z, f))
        self.frame[:, qubit] = new_f
        return outcome

    def measure_x(self, qubit: int) -> np.ndarray:
        """X-basis measurement. Outcome is 1 iff frame has Z or Y."""
        f = self.frame[:, qubit]
        outcome = (f == Z) | (f == Y)
        new_f = np.where(f == Z, I, np.where(f == Y, X, f))
        self.frame[:, qubit] = new_f
        return outcome

    def copy(self) -> 'PauliFrame':
        pf = PauliFrame(self.n_shots, self.n_qubits)
        pf.frame = self.frame.copy()
        return pf
