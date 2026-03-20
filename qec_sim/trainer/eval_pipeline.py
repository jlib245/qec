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
    def __init__(self, config_path: str, model_path: str = None):
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
        log_dir = os.path.dirname(self.model_path) if self.model_path else "."
        log_path = os.path.join(log_dir, f"eval_{timestamp}.log")
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

    def _build_neural_decoder(self):
        if self.model_path is None:
            raise ValueError("neural_decoder 사용 시 --model 경로가 필요합니다.")
        _, wrapped_model = ComponentFactory.build_system(self.config)
        state = torch.load(self.model_path, map_location=self.device)
        wrapped_model.load_state_dict(state)
        wrapped_model = wrapped_model.to(self.device)
        wrapped_model.eval()
        return NeuralDecoder(model=wrapped_model)

    def _run(self, shots: int, timestamp: str):
        decoder_name = self.config.decoder.name
        print(f"평가 시작 (Device: {self.device}, shots/noise: {shots:,})")
        print(f"디코더: {decoder_name}")
        if self.model_path:
            print(f"모델: {self.model_path}")
        print()

        # 1. neural_decoder는 모델을 미리 로드
        neural_decoder = None
        if decoder_name == "neural_decoder":
            neural_decoder = self._build_neural_decoder()

        # 2. 노이즈별 평가
        results = []
        noise_configs = self.config.get_expanded_noise_configs()
        model_dir = os.path.dirname(self.model_path) if self.model_path else "."
        ler_label = "LER"
        print(f"{'p_gate':>8} {'p_meas':>8} {'p_leak':>8} | {ler_label:>12}")
        print("-" * 45)

        for noise_cfg in noise_configs:
            circuit = build_circuit(
                self.config.code.name, self.config.code, noise_cfg
            ).build()
            simulator = CircuitNoiseSimulator(circuit, noise_cfg)

            raw = simulator.generate_data(shots=shots)
            syndromes, observables, erasures = raw['syndromes'], raw['observables'], raw['erasures']

            if decoder_name == "neural_decoder":
                preds = neural_decoder.decode_batch(
                    syndromes,
                    erasures=erasures if self.config.model.use_erasures else None,
                    batch_size=4096,
                )
            elif decoder_name == "mwpm":
                dem = circuit.detector_error_model(decompose_errors=True)
                mwpm = ErasureMWPM(error_model=dem)
                preds = mwpm.decode_batch(syndromes, erasures=erasures)
            else:
                raise ValueError(f"지원하지 않는 decoder: {decoder_name}")

            ler = float(np.mean(np.any(preds != observables, axis=1)))

            row = {
                "p_gate":      noise_cfg.p_gate,
                "p_meas":      noise_cfg.p_meas,
                "p_corr":      noise_cfg.p_corr,
                "p_leak":      noise_cfg.p_leak,
                "shots":       shots,
                "decoder":     decoder_name,
                "ler":         ler,
            }
            results.append(row)
            print(f"{noise_cfg.p_gate:>8.4f} {noise_cfg.p_meas:>8.4f} {noise_cfg.p_leak:>8.4f} "
                  f"| {ler:>12.4%}")

        # 3. 저장
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
