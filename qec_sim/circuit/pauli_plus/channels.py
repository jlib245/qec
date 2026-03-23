# qec_sim/circuit/pauli_plus/channels.py
"""
Noise channels operating on PauliFrame.
Phase 1: depolarizing (1q and 2q), bit/phase flip.
"""

import numpy as np
from .frame import PauliFrame, I, X, Y, Z

RNG = np.random.default_rng


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def depolarize1(frame: PauliFrame, qubit: int, p: float, rng=None,
                mask: np.ndarray | None = None):
    """
    Single-qubit depolarizing: X/Y/Z each with prob p/3.
    mask: (n_shots,) bool — None이면 전체 shots에 적용.
    """
    if p <= 0:
        return
    rng = _rng(rng)
    n = frame.n_shots
    r = rng.random(n)
    pauli = np.where(r < p / 3, X,
            np.where(r < 2 * p / 3, Y,
            np.where(r < p, Z, I))).astype(np.int8)
    if mask is not None:
        pauli[~mask] = I
    frame.apply_single_pauli(qubit, pauli)


def depolarize2(frame: PauliFrame, q0: int, q1: int, p: float, rng=None):
    """
    Two-qubit depolarizing: 15 non-identity Pauli pairs, each with prob p/15.
    """
    if p <= 0:
        return
    rng = _rng(rng)
    # All 16 2q Paulis except II
    paulis = [(a, b) for a in range(4) for b in range(4) if not (a == 0 and b == 0)]
    r = rng.integers(0, 15, size=frame.n_shots)
    mask = rng.random(frame.n_shots) < p
    idx = r[mask]
    p0 = np.array([paulis[i][0] for i in idx], dtype=np.int8)
    p1 = np.array([paulis[i][1] for i in idx], dtype=np.int8)

    f0 = frame.frame[mask, q0]
    f1 = frame.frame[mask, q1]
    from .frame import PAULI_MUL
    # leaked 큐빗(state >= 4)은 Pauli 연산 스킵 — leakage subspace에 Pauli 미작용
    leaked0 = f0 >= 4
    leaked1 = f1 >= 4
    new_f0 = PAULI_MUL[np.where(leaked0, 0, f0), p0]
    new_f1 = PAULI_MUL[np.where(leaked1, 0, f1), p1]
    frame.frame[mask, q0] = np.where(leaked0, f0, new_f0)
    frame.frame[mask, q1] = np.where(leaked1, f1, new_f1)


def bitflip(frame: PauliFrame, qubit: int, p: float, rng=None):
    if p <= 0:
        return
    rng = _rng(rng)
    err = (rng.random(frame.n_shots) < p).astype(np.int8) * X
    frame.apply_single_pauli(qubit, err)


def phaseflip(frame: PauliFrame, qubit: int, p: float, rng=None):
    if p <= 0:
        return
    rng = _rng(rng)
    err = (rng.random(frame.n_shots) < p).astype(np.int8) * Z
    frame.apply_single_pauli(qubit, err)
