"""GARI-NMS decoder (Maan et al., arXiv:2510.14060).

Thin BaseDecoder adapter around the C decoder + Python helpers in
`external/gari-nms`. Z-basis single-memory experiment only (matches their
`stim_batched_data_v2.py` driver). Uses the `'ibm'` variant
(`filter_detectors_by_basis`) for detector-type classification, which is
generic across BB circuits.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import stim

from .base import BaseDecoder
from .registry import register_decoder

_GARI_PATH = Path(__file__).resolve().parents[2] / "external" / "gari-nms"
if not _GARI_PATH.exists():
    raise ImportError(
        f"GARI submodule missing at {_GARI_PATH}. "
        "Run: git submodule update --init external/gari-nms"
    )
if not (_GARI_PATH / "my_decoders" / "hbplib_v2.so").exists():
    raise ImportError(
        f"GARI .so not built. Run: cd {_GARI_PATH}/my_decoders && make"
    )


@contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# Upstream wrapper does `ctypes.CDLL('my_decoders/hbplib_v2.so')` at module
# import time, so we need cwd=gari-nms root for the import.
sys.path.insert(0, str(_GARI_PATH))
with _cwd(_GARI_PATH):
    from helper_functions import (  # noqa: E402
        build_Hcols_Hrows,
        det_and_err_encoding,
    )
    from my_decoders.hbplib_wrapper_v2 import (  # noqa: E402
        ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler as _gari_decode,
    )

from beliefmatching import detector_error_model_to_check_matrices  # noqa: E402


@register_decoder("gari")
class GariNmsDecoder(BaseDecoder):
    def __init__(
        self,
        circuit: stim.Circuit,
        error_model: Optional[stim.DetectorErrorModel] = None,
        max_iter: int = 400,
        msf: float = 0.96875,
        ensemble: int = 1,
        prob_or_itr: str = "i",
        seed: int = 1,
        custom_schedule: int = 2,
        prior_type: int = 0,
        early_stopping: bool = True,
        num_threads: int = 1,
        **kwargs,
    ):
        # BB DEMs aren't graphlike-decomposable; skip the surface-code-flavored
        # fault_id → qubit map by passing circuit=None to super.
        super().__init__(circuit=None)
        if error_model is None:
            error_model = circuit.detector_error_model(
                decompose_errors=False,
                flatten_loops=True,
                ignore_decomposition_failures=True,
            )
        self._matrices = detector_error_model_to_check_matrices(
            error_model, allow_undecomposed_hyperedges=True
        )
        # `d` is only used by det_and_err_encoding for the parity check; the
        # 'ibm' variant ignores its value but requires d % 2 == 0.
        self._all_xz = det_and_err_encoding(
            d=2, matrices=self._matrices, circuit=circuit, variant="ibm"
        )
        big_matrix_dense = self._all_xz.big_matrix.toarray()
        self._big_shape = big_matrix_dense.shape
        self._Hcols, self._Hrows = build_Hcols_Hrows(big_matrix_dense)

        m, n = self._matrices.check_matrix.shape
        mx, nx = self._all_xz.dx.shape
        mz, nz = self._all_xz.dz.shape
        self._m, self._mx, self._mz = m, mx, mz
        self._nx, self._nz = nx, nz
        self._h_shape = (m, n)

        det_index = self._all_xz.det_index
        self._idx_x_det = np.where(det_index == 1)[0]  # X-detectors
        self._idx_z_det = np.where(det_index == 3)[0]  # Z-detectors

        priors = self._matrices.priors
        priors_ab = np.zeros(nx + nz, dtype=float)
        for i in range(nx):
            priors_ab[i] = priors[np.where(self._all_xz.i_dx_in_hx == i)[0]].sum()
        for i in range(nz):
            priors_ab[nx + i] = priors[np.where(self._all_xz.i_dz_in_hz == i)[0]].sum()
        self._llr_ab = np.log((1.0 - priors_ab) / np.clip(priors_ab, 1e-300, None))

        priors_big = np.zeros(self._big_shape[1], dtype=float)
        if prior_type == 0:
            priors_big[: nx + nz] = 0.5
        elif prior_type == 2:
            priors_big[: nx + nz] = priors_ab
        else:
            raise ValueError(f"prior_type must be 0 or 2, got {prior_type}")
        priors_big[nx + nz : 2 * nx + nz] = priors[self._all_xz.i_hx_only]
        priors_big[2 * nx + nz : 2 * (nx + nz)] = priors[self._all_xz.i_hz_only]
        priors_big[2 * (nx + nz) :] = priors[self._all_xz.i_hy_only]
        self._llr = np.log((1.0 - priors_big) / np.clip(priors_big, 1e-300, None))

        l = self._matrices.observables_matrix.toarray().astype(np.int8)
        self._l_dz = l[:, self._all_xz.i_hz_only]
        self._dz = self._all_xz.dz.astype(np.int8)

        self._max_iter = max_iter
        self._msf = msf
        self._ensemble = max(ensemble, 1)
        self._poi = prob_or_itr
        self._seed = seed
        self._cs = custom_schedule
        self._early = early_stopping
        self._num_threads = num_threads

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        batch = syndromes.shape[0]
        det_big = np.zeros((batch, self._big_shape[0]), dtype=np.int8)
        det_big[:, : self._mx] = syndromes[:, self._idx_x_det].astype(np.int8)
        det_big[:, self._mx : self._m] = syndromes[:, self._idx_z_det].astype(np.int8)
        # bipolar encoding {0,1} -> {1,-1}
        det_bipolar = (1 - 2 * det_big).astype(np.int8)

        logical_out, _iters = _gari_decode(
            self._llr, self._llr_ab, batch, self._max_iter,
            self._Hrows, self._Hcols,
            self._msf, det_bipolar,
            self._seed, self._h_shape, self._all_xz.dx.shape,
            self._ensemble, self._poi,
            self._nx, self._nz, self._mx, self._m,
            self._dz, self._l_dz,
            self._cs, early_stopping=bool(self._early),
            num_threads=self._num_threads,
        )
        return logical_out.astype(bool)

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        preds = self.decode_batch(syndrome[None, :])
        return {
            "logical_error": preds[0].tolist(),
            "corrected_fault_ids": [],
        }
