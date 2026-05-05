# qec_sim/data/generator.py
import numpy as np
from pathlib import Path
import time

from qec_sim.config.schema import CodeParams, NoiseParams, ExperimentConfig
from qec_sim.circuit.simulator import SimulatorPool


class DatasetGenerator:
    """Stim 또는 Pauli+ 시뮬레이터 풀에서 오프라인 데이터셋을 생성·저장.

    이전 버전이 Stim-only이고 모든 키를 int8로 캐스트했던 버그를 수정.
    soft_measurements 같은 float 데이터를 native dtype으로 보존.
    """

    def __init__(self, simulator_pool: SimulatorPool):
        self.pool = simulator_pool

    # ------------------------------------------------------------------ #
    # Constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "DatasetGenerator":
        """ExperimentConfig에서 simulation.backend에 따라 적절한 풀 빌드."""
        from qec_sim.trainer.factory import _build_simulator_pool
        return cls(_build_simulator_pool(config))

    @classmethod
    def from_stim(cls, code_config: CodeParams,
                  noise_configs: list[NoiseParams]) -> "DatasetGenerator":
        return cls(SimulatorPool.from_stim(code_config, noise_configs))

    @classmethod
    def from_pauli_plus(cls, code_config: CodeParams,
                        noise_configs: list) -> "DatasetGenerator":
        from qec_sim.circuit.pauli_plus import PauliPlusSimulator
        sims = [PauliPlusSimulator(code_config, n) for n in noise_configs]
        return cls(SimulatorPool(sims))

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate_and_save(self, shots: int, save_dir: str, filename: str,
                          batch_size: int, compress: bool = True):
        from tqdm.auto import tqdm
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / f"{filename}.npz"

        simulators = self.pool.get_all_simulators()
        num_configs = len(simulators)
        print(f"[{filename}] 총 {shots:,}샷 생성을 시작 ({num_configs}개의 노이즈 환경)", flush=True)
        start_time = time.time()

        # 1샷 probe 결과로 버퍼 사전 할당 (이전 버전이 모든 키를 int8로 캐스트해
        # soft IQ float이 손상됐던 버그 방지 — native dtype/shape를 시뮬에서 직접 추론).
        # IQ off일 때 soft_measurements 같은 None 값은 키에서 제외.
        probe = simulators[0].generate_data(shots=1)
        present_keys = [k for k, v in probe.items() if isinstance(v, np.ndarray)]

        buffers = {
            k: np.zeros((shots,) + probe[k].shape[1:], dtype=probe[k].dtype)
            for k in present_keys
        }

        base_shots = shots // num_configs
        remainder = shots % num_configs
        generated_count = 0

        with tqdm(total=shots, unit='shot', desc=f'gen {filename}', dynamic_ncols=True) as pbar:
            for i, simulator in enumerate(simulators):
                config_shots = base_shots + (1 if i < remainder else 0)
                if config_shots == 0:
                    continue

                config_generated = 0
                while config_generated < config_shots:
                    current_shots = min(config_shots - config_generated, batch_size)
                    raw = simulator.generate_data(shots=current_shots)

                    end_idx = generated_count + current_shots
                    for k in buffers:
                        buffers[k][generated_count:end_idx] = raw[k]

                    generated_count = end_idx
                    config_generated += current_shots
                    pbar.update(current_shots)

        print("  -> 셔플 + 저장 중...", flush=True)
        indices = np.random.permutation(shots)
        shuffled = {k: v[indices] for k, v in buffers.items()}

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(filepath, **shuffled)

        size_mb = filepath.stat().st_size / 1e6
        print(f"✅ 저장: {filepath} ({size_mb:.0f} MB, {time.time() - start_time:.1f}초)\n", flush=True)
