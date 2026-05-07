"""sinter용 generic neural decoder adapter.

학습 결과 디렉터리(`run_dir`)만 받으면 yaml의 `model.name`/`coset_mode` 등을
보고 회로/모델/전처리기/coset LUT를 자동 구성. QCT, JungCNN, MLP 등 모든
`REQUIRED_PREPROCESSOR`를 따르는 모델에 동일하게 적용.

설계 포인트
- `__init__`은 path와 옵션 정도만 보관해서 pickle 가벼움 (sinter 멀티워커가
  매 worker마다 디스크에서 새로 로드하는 게 더 깔끔).
- 무거운 build/load는 첫 `compile_decoder_for_dem` 호출 시 1회 수행.
- Stim 백엔드만 지원 — Pauli+ 시뮬레이터는 sinter sampling 불가.

⚠️ 성능 주의 (2026-05 기준 미해결)
sinter는 streaming 패턴으로 작은 batch (수~수백 shot)를 자주 보내는데, PyTorch
eager 모드의 호출당 오버헤드가 transformer에 누적되어 small-batch에서 매우 느려짐
(d=5 실측 ~1.6 s/shot vs batch=1024 시 4 ms/shot, 400× 격차). 풀려면 `torch.compile`
또는 batch 버퍼링 layer 도입 필요. 현재는 NeuralSinterDecoder를 sinter benchmark에
넣지 않는 것을 권장 — neural decoder는 우리 yaml eval pipeline에서 GPU batch
추론으로 평가하고, sinter는 알고리즘 디코더(MWPM/BM/BP+OSD)에만 사용.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import sinter
import stim
import torch

from qec_sim.decoders.lut import (
    build_detector_lut,
    coset_index_to_bits,
    compute_lut_correction,
)


class _NeuralCompiled(sinter.CompiledDecoder):
    def __init__(
        self,
        wrapper: torch.nn.Module,
        coset_lut: Optional[np.ndarray],
        num_detectors: int,
        num_observables: int,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.wrapper = wrapper
        self.coset_lut = coset_lut  # None이면 binary 모드
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.device = device
        self.batch_size = batch_size
        self._num_obs_bytes = -(-num_observables // 8)

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        N = bit_packed_detection_event_data.shape[0]
        unpacked = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self.num_detectors,
            bitorder="little",
        )

        all_obs = []
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            batch_syn = unpacked[s:e]
            batch_dict = {
                "syndromes": torch.from_numpy(batch_syn).to(self.device, dtype=torch.float32)
            }
            with torch.no_grad():
                logits = self.wrapper(batch_dict)

            if self.coset_lut is not None:
                # coset 모드: argmax → coset bits → LUT correction과 XOR
                coset_pred = logits.argmax(dim=-1).cpu().numpy()
                coset_bits = coset_index_to_bits(coset_pred, self.num_observables)
                lut_corr = compute_lut_correction(batch_syn, self.coset_lut)
                obs_pred = (coset_bits ^ lut_corr).astype(np.uint8)
            else:
                # binary 모드: per-observable sigmoid threshold
                obs_pred = (logits > 0).cpu().numpy().astype(np.uint8)

            all_obs.append(obs_pred)

        preds = np.concatenate(all_obs, axis=0)
        packed = np.packbits(preds, axis=1, bitorder="little")
        return packed[:, : self._num_obs_bytes].astype(np.uint8, copy=False)


class NeuralSinterDecoder(sinter.Decoder):
    """모든 qec_sim neural model을 sinter Decoder 인터페이스로 노출.

    Usage:
        dec = NeuralSinterDecoder("results/.../qct_si1000_d5", device="cpu")
        sinter.collect(custom_decoders={"qct": dec}, ...)

    같은 `run_dir`에서 만든 인스턴스는 한 토폴로지(같은 distance/rounds/code)에
    대응 — 다른 distance를 같이 평가하려면 각각 따로 만들어 다른 이름으로 등록.
    """

    def __init__(
        self,
        run_dir: str | pathlib.Path,
        device: str = "cpu",
        batch_size: int = 4096,
    ) -> None:
        self.run_dir = str(pathlib.Path(run_dir))
        self.device_str = device
        self.batch_size = batch_size
        self._compiled: Optional[_NeuralCompiled] = None

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        if self._compiled is not None:
            return self._compiled

        # 지연 import — 모듈 로드 자체는 가볍게 유지
        from qec_sim.config.schema import ExperimentConfig, NoiseParams
        from qec_sim.circuit.registry import build_circuit
        from qec_sim.trainer.factory import ComponentFactory

        run_dir = pathlib.Path(self.run_dir)
        cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
        if cfg.simulation.backend != "stim":
            raise ValueError(
                f"sinter는 stim 백엔드 전용 (config.simulation.backend={cfg.simulation.backend})"
            )

        # Factory가 simulator_pool도 같이 빌드해서 좀 무겁지만, DRY 우선 (worker당 1회).
        _datamodule, wrapper = ComponentFactory.build_system(cfg)

        device = torch.device(self.device_str)
        state = torch.load(run_dir / "best_model.pth", map_location=device)
        wrapper.to(device)
        wrapper.load_state_dict(state)
        wrapper.eval()

        # coset_lut과 회로 메타는 factory 내부에서만 쓰여 외부로 안 나오므로 재빌드.
        rep_p = (
            cfg.noise.p_gate[0]
            if isinstance(cfg.noise.p_gate, list)
            else cfg.noise.p_gate
        )
        rep_meas = (
            cfg.noise.p_meas[0]
            if isinstance(cfg.noise.p_meas, list)
            else cfg.noise.p_meas
        )
        circuit = build_circuit(
            cfg.code.name,
            cfg.code,
            NoiseParams(p_gate=rep_p, p_meas=rep_meas, p_corr=0.0),
        ).build()

        if dem.num_detectors != circuit.num_detectors:
            raise ValueError(
                f"DEM detectors ({dem.num_detectors}) ≠ run_dir circuit "
                f"detectors ({circuit.num_detectors}) — distance mismatch?"
            )

        coset_lut = build_detector_lut(circuit) if cfg.model.coset_mode else None

        self._compiled = _NeuralCompiled(
            wrapper=wrapper,
            coset_lut=coset_lut,
            num_detectors=circuit.num_detectors,
            num_observables=circuit.num_observables,
            device=device,
            batch_size=self.batch_size,
        )
        return self._compiled
