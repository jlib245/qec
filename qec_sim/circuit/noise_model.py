"""Circuit-level noise models, vendored from Gidney's honeycomb_threshold.

Source: https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py
License: Apache-2.0 (see upstream LICENSE file)
Author: Craig Gidney (Google Quantum AI)
Date imported: 2026-05-11

Provides exact SI1000, SD6, PC3, EM3 noise model implementations as defined
in "A Fault-Tolerant Honeycomb Memory" (Gidney et al., 2021).

We use this for **canonical SI1000 noise injection** on top of zero-noise
stim-generated circuits — to ensure parity with Higgott-Gidney 2023, Gidney
2023 (yoked SC), and other reference benchmarks.

SI1000 prescription:
  any_clifford_1     = p / 10    (1-qubit Clifford)
  any_clifford_2     = p         (2-qubit Clifford — CX/CZ/etc.)
  idle               = p / 10    (qubits idle during 2q gates)
  measure_reset_idle = 2p        (data qubit idle during ancilla M/R)
  noisy_gates: R → 2p, M → 5p

Note: We add "CX" to the canonical noisy_gates so that stim's default
surface_code:rotated_memory_z generator (which uses CX rather than CZ) is
noised consistently. CX and CZ are both 2-qubit Cliffords, and SI1000's
prescription treats all 2q gates identically (= p).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Set, Tuple

import stim

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    idle: float
    measure_reset_idle: float
    noisy_gates: Dict[str, float]
    any_clifford_1: Optional[float] = None
    any_clifford_2: Optional[float] = None
    use_correlated_parity_measurement_errors: bool = False

    @staticmethod
    def SI1000(p: float) -> "NoiseModel":
        """Canonical SI1000 (Gidney 2021). 2q gates p, 1q p/10, M=5p, R=2p."""
        return NoiseModel(
            any_clifford_1=p / 10,
            any_clifford_2=p,
            idle=p / 10,
            measure_reset_idle=2 * p,
            noisy_gates={
                "CX": p, "CY": p, "CZ": p,
                "R": 2 * p, "RX": 2 * p, "RY": 2 * p,
                "M": 5 * p, "MX": 5 * p, "MY": 5 * p,
                "MR": 5 * p, "MRX": 5 * p, "MRY": 5 * p,
            },
        )

    def apply(self, base_circuit: stim.Circuit) -> stim.Circuit:
        """Apply this noise model to a zero-noise stim circuit. Convenience alias."""
        return self.noisy_circuit(base_circuit)

    @staticmethod
    def SD6(p: float) -> "NoiseModel":
        return NoiseModel(
            any_clifford_1=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={"CX": p, "R": p, "M": p, "MR": p},
        )

    def noisy_op(
        self, op: stim.CircuitInstruction, p: float, ancilla: int
    ) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                post.append_operation("DEPOLARIZE1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                post.append_operation("DEPOLARIZE2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    post.append_operation(
                        "Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
                if op.name in MEASURE_OPS:
                    pre.append_operation(
                        "Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
            else:
                raise NotImplementedError(repr(op))
        mid.append_operation(op.name, targets, args)
        return pre, mid, post

    def noisy_circuit(
        self, circuit: stim.Circuit, *, qs: Optional[Set[int]] = None
    ) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits

        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        used_qubits: Set[int] = set()
        measured_or_reset_qubits: Set[int] = set()
        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid:
                return

            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append_operation(
                    "DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_or_reset_qubits)
            if measured_or_reset_qubits and idle_qubits and self.measure_reset_idle > 0:
                current_moment_post.append_operation(
                    "DEPOLARIZE1", idle_qubits, self.measure_reset_idle)

            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_or_reset_qubits.clear()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                flush()
                result += self.noisy_circuit(op.body_copy(), qs=qs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    flush()
                    result.append_operation("TICK", [])
                    continue

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    p = 0
                else:
                    raise NotImplementedError(repr(op))
                pre, mid, post = self.noisy_op(op, p, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                touched_qubits = {
                    t.value
                    for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in ANNOTATION_OPS:
                    touched_qubits.clear()
                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()

        return result
