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


def depolarize1(frame: PauliFrame, qubit: int, p: float, rng=None):
    """Single-qubit depolarizing: applies X/Y/Z each with prob p/3."""
    if p <= 0:
        return
    rng = _rng(rng)
    r = rng.random(frame.n_shots)
    pauli = np.where(r < p / 3, X,
            np.where(r < 2 * p / 3, Y,
            np.where(r < p, Z, I))).astype(np.int8)
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
    frame.frame[mask, q0] = PAULI_MUL[f0, p0]
    frame.frame[mask, q1] = PAULI_MUL[f1, p1]


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
