"""Pauli+ 시뮬레이터의 SI1000 baseline 거동이 Stim과 일치.

두 버그(CX transpose + 수동 detector 파싱)는 둘 다
'Pauli+ LER이 Stim 대비 17-543× 부풀어있다'는 형태로 드러났음.
회귀 방지용 — 이 테스트가 깨지면 둘 중 하나가 다시 망가졌다는 신호.

빠른 실행을 위해 d=3 + p=1e-3 + 5000 shots로 제한 (수 초 내).
"""
import numpy as np
import pytest
import stim
import pymatching

from qec_sim.config.schema import CodeParams, PauliPlusNoiseParams
from qec_sim.circuit.pauli_plus import PauliPlusSimulator


def test_pauli_plus_d3_baseline_matches_stim():
    """d=3, p=1e-3, full SI1000 4채널: Pauli+ LER ≈ Stim self-consistent LER."""
    d = 3
    p = 1e-3
    shots = 5000  # 빠른 검증용
    rng_seed = 42

    # Pauli+ 데이터 생성
    sim = PauliPlusSimulator(
        CodeParams(name="surface_code", distance=d, rounds=d),
        PauliPlusNoiseParams(p=p),  # 비-Pauli 효과 모두 0
        seed=rng_seed,
    )
    raw = sim.generate_data(shots=shots)

    # Stim 풀 SI1000 4채널 회로 → MWPM DEM
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=2 * p,
        before_measure_flip_probability=5 * p,
        after_reset_flip_probability=2 * p,
    )
    matcher = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True)
    )
    pp_ler = float(np.mean(np.any(
        matcher.decode_batch(raw["syndromes"]) != raw["observables"], axis=1
    )))

    # Stim 자기 sampling (참조 baseline)
    s, o = circ.compile_detector_sampler(seed=rng_seed).sample(
        shots, separate_observables=True
    )
    stim_ler = float(np.mean(np.any(matcher.decode_batch(s) != o, axis=1)))

    # 통계 오차: σ ≈ √(p(1-p)/N), p≈0.002라 ~0.06% — 5σ 허용으로 0.3% 차이 한도.
    # 버그 있을 때 격차는 17×+ (Pauli+ 3.8% vs Stim 0.2%) 라 확실히 잡힘.
    assert abs(pp_ler - stim_ler) < 0.005, (
        f"Pauli+ LER ({pp_ler:.4%}) and Stim LER ({stim_ler:.4%}) diverge — "
        f"CX 테이블 또는 detector 파싱 회귀 의심"
    )


def test_pauli_plus_zero_noise_zero_syndromes():
    """p=0이면 모든 신드롬 0 (가장 기본 sanity)."""
    sim = PauliPlusSimulator(
        CodeParams(name="surface_code", distance=3, rounds=3),
        PauliPlusNoiseParams(p=0.0),
    )
    raw = sim.generate_data(shots=100)
    assert not np.any(raw["syndromes"])
    assert not np.any(raw["observables"])
