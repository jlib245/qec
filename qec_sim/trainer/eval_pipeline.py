# qec_sim/trainer/eval_pipeline.py
import os
import csv
import datetime
import torch
import numpy as np

from qec_sim.config.schema import ExperimentConfig
from qec_sim.trainer.factory import ComponentFactory
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.decoders.mwpm import ErasureMWPM
from qec_sim.decoders.neural import NeuralDecoder


class EvaluationPipeline:
    """
    학습된 모델을 MWPM baseline과 비교 평가합니다.

    각 노이즈 설정마다:
      - MWPM (erasure 인지)
      - Neural decoder (erasure 인지)
    의 LER을 측정하고 CSV로 저장합니다.
    """

    def __init__(self, config_path: str, model_path: str):
        self.config = ExperimentConfig.from_yaml(config_path)
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def run(self, shots: int = 10000):
        # eval 로그를 모델 디렉토리에 파일로도 저장
        import sys
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.dirname(self.model_path)
        log_path = os.path.join(model_dir, f"eval_{timestamp}.log")
        log_file = open(log_path, 'w', encoding='utf-8')

        # 터미널 + 파일 동시 출력 (TrainingPipeline의 _Tee와 동일한 방식)
        class _Tee:
            def __init__(self, *streams): self._s = streams
            def write(self, d): [s.write(d) for s in self._s]
            def flush(self): [s.flush() for s in self._s]

        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout, log_file)

        try:
            self._run(shots, timestamp)
        finally:
            sys.stdout = orig_stdout
            log_file.close()

    def _run(self, shots: int, timestamp: str):
        print(f"평가 시작 (Device: {self.device}, shots/noise: {shots:,})")
        print(f"모델: {self.model_path}\n")

        # 1. 모델 로드
        _, wrapped_model = ComponentFactory.build_system(self.config)
        state = torch.load(self.model_path, map_location=self.device)
        wrapped_model.load_state_dict(state)
        wrapped_model = wrapped_model.to(self.device)
        wrapped_model.eval()

        neural_decoder = NeuralDecoder(model=wrapped_model)

        # 2. 노이즈별 평가
        results = []
        noise_configs = self.config.get_expanded_noise_configs()
        model_dir = os.path.dirname(self.model_path)
        print(f"{'p_gate':>8} {'p_meas':>8} {'p_leak':>8} | {'MWPM LER':>10} {'Neural LER':>12} {'개선율':>8}")
        print("-" * 65)

        for noise_cfg in noise_configs:
            circuit = build_circuit(
                self.config.code.name, self.config.code, noise_cfg
            ).build()
            simulator = CircuitNoiseSimulator(circuit, noise_cfg)
            dem = circuit.detector_error_model(decompose_errors=True)
            mwpm = ErasureMWPM(error_model=dem)

            raw = simulator.generate_data(shots=shots)
            syndromes, observables, erasures = raw['syndromes'], raw['observables'], raw['erasures']

            # MWPM (erasure 정보 활용)
            mwpm_preds = mwpm.decode_batch(syndromes, erasures=erasures)
            mwpm_ler = float(np.mean(np.any(mwpm_preds != observables, axis=1)))

            # Neural decoder
            neural_preds = neural_decoder.decode_batch(syndromes, erasures=erasures)
            neural_ler = float(np.mean(np.any(neural_preds != observables, axis=1)))

            improvement = (mwpm_ler - neural_ler) / mwpm_ler * 100 if mwpm_ler > 0 else 0.0

            row = {
                "p_gate":      noise_cfg.p_gate,
                "p_meas":      noise_cfg.p_meas,
                "p_corr":      noise_cfg.p_corr,
                "p_leak":      noise_cfg.p_leak,
                "shots":       shots,
                "mwpm_ler":    mwpm_ler,
                "neural_ler":  neural_ler,
                "improvement": improvement,
            }
            results.append(row)
            print(f"{noise_cfg.p_gate:>8.4f} {noise_cfg.p_meas:>8.4f} {noise_cfg.p_leak:>8.4f} "
                  f"| {mwpm_ler:>10.4%} {neural_ler:>12.4%} {improvement:>+7.1f}%")

        # 3. CSV 저장
        self._save_results(results, model_dir, timestamp)

    def _save_results(self, results: list, model_dir: str, timestamp: str):
        save_path = os.path.join(model_dir, f"eval_{timestamp}.csv")

        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        log_path = os.path.join(model_dir, f"eval_{timestamp}.log")
        print(f"\n평가 완료.")
        print(f"  CSV: {save_path}")
        print(f"  LOG: {log_path}")
