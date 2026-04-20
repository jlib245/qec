# qec_sim/decoders/lut.py
"""
LUT(Look-Up Table) 기반 coset decoder.

Stim DEM에서 per-detector correction을 추출하여,
detection events → base correction → coset label (4-class) 변환을 수행.

Coset classes:
  0 = II (no logical error)
  1 = IX (X logical flip only)
  2 = ZI (Z logical flip only)
  3 = ZX (both X and Z = Y logical)
"""
import numpy as np
import stim


def build_detector_lut(circuit: stim.Circuit) -> np.ndarray:
    """
    Stim circuit의 DEM에서 per-detector LUT를 구성.

    각 detector에 대해, 해당 detector를 트리거하는 에러 메커니즘 중
    하나를 선택하여 observable flip 패턴을 기록.

    Args:
        circuit: Stim circuit

    Returns:
        lut: (num_detectors, num_observables) uint8 배열.
             lut[d] = detector d가 발화했을 때의 canonical observable flip.
    """
    num_detectors = circuit.num_detectors
    num_observables = circuit.num_observables
    lut = np.zeros((num_detectors, num_observables), dtype=np.uint8)
    assigned = np.zeros(num_detectors, dtype=bool)

    dem = circuit.detector_error_model(decompose_errors=True)

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue

        dets = []
        obs = np.zeros(num_observables, dtype=np.uint8)
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                dets.append(target.val)
            elif target.is_logical_observable_id():
                obs[target.val] = 1

        for d in dets:
            if not assigned[d]:
                lut[d] = obs
                assigned[d] = True

    return lut


def compute_lut_correction(syndromes: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Detection events에 LUT를 적용하여 base correction 계산.

    Args:
        syndromes: (N, num_detectors) bool/uint8
        lut: (num_detectors, num_observables) uint8

    Returns:
        corrections: (N, num_observables) uint8
    """
    # syndromes @ lut → 각 observable에 대한 flip 횟수 → mod 2
    corrections = (syndromes.astype(np.uint8) @ lut) % 2
    return corrections.astype(np.uint8)


def compute_coset_labels(syndromes: np.ndarray, observables: np.ndarray,
                         lut: np.ndarray) -> np.ndarray:
    """
    Detection events + actual observables + LUT → coset class (0~3).

    Args:
        syndromes: (N, num_detectors) bool/uint8
        observables: (N, num_observables) bool/uint8
        lut: (num_detectors, num_observables) uint8

    Returns:
        labels: (N,) int64, values in {0, 1, 2, 3}
    """
    corrections = compute_lut_correction(syndromes, lut)
    coset_bits = (corrections ^ observables.astype(np.uint8))  # (N, num_obs)

    # observable 수에 따라 class 수 결정
    # 1 observable → 2-class (0, 1)
    # 2 observables → 4-class (obs[1]*2 + obs[0])
    n_obs = coset_bits.shape[1]
    if n_obs == 1:
        labels = coset_bits[:, 0].astype(np.int64)
    else:
        labels = coset_bits[:, 1].astype(np.int64) * 2 + coset_bits[:, 0].astype(np.int64)

    return labels


def coset_index_to_bits(indices: np.ndarray, n_obs: int) -> np.ndarray:
    """
    Coset class index → observable bit 배열로 변환.
    compute_coset_labels의 역연산.

    Args:
        indices: (N,) int, class indices
        n_obs: observable 수

    Returns:
        bits: (N, n_obs) uint8
    """
    if n_obs == 1:
        return indices[:, np.newaxis].astype(np.uint8)
    else:
        obs0 = (indices % 2).astype(np.uint8)
        obs1 = (indices // 2).astype(np.uint8)
        return np.stack([obs0, obs1], axis=-1)
