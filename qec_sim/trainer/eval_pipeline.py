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
from qec_sim.decoders.mwpm import MWPMDecoder
from qec_sim.decoders.neural import NeuralDecoder
from qec_sim.decoders.belief_matching import BeliefMatchingDecoder
from qec_sim.decoders.belief_matching_fast import BeliefMatchingFastDecoder
from qec_sim.decoders.bp_osd import BpOsdDecoder


class EvaluationPipeline:
    def __init__(self, config_path: str, model_path: str = None):
        self.config = ExperimentConfig.from_yaml(config_path)
        self.config_path = config_path
        self.model_path = model_path
        from qec_sim.core.interfaces import get_best_device
        self.device = get_best_device()

    def _resolve_output_dir(self, timestamp: str) -> str:
        # 로컬 model_path 가 있으면 그 옆에 (학습 디렉토리 재사용),
        # registry/run URI 거나 model_path 없으면 새 timestamped 디렉토리.
        is_uri = isinstance(self.model_path, str) and self.model_path.startswith(("models:/", "runs:/"))
        if self.model_path and not is_uri:
            return os.path.dirname(self.model_path)
        from qec_sim.trainer.utils import timestamped_output_dir
        root = timestamped_output_dir(self.config.training.output_dir, timestamp)
        os.makedirs(root, exist_ok=True)
        return root

    def run(self, shots: int):
        import sys
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self._resolve_output_dir(timestamp)
        log_path = os.path.join(log_dir, f"eval_{timestamp}.log")
        log_file = open(log_path, 'w', encoding='utf-8')

        from qec_sim.trainer.callbacks import _Tee
        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout, log_file)

        try:
            self._run(shots, timestamp)
        except BaseException:
            # 활성 mlflow run 이 있으면 FAILED 로 종료 (artifact 일부라도 보존).
            try:
                import mlflow as _mlflow
                if _mlflow.active_run() is not None:
                    _mlflow.end_run(status="FAILED")
            except Exception:
                pass
            raise
        finally:
            sys.stdout = orig_stdout
            log_file.close()

    def _build_neural_decoder(self, circuit=None):
        if self.model_path is None:
            raise ValueError("neural_decoder 사용 시 --model 경로가 필요합니다.")
        _, wrapped_model = ComponentFactory.build_system(self.config)

        # URI 스킴 기반 분기: registry/run artifact 면 mlflow 로, 아니면 torch.load.
        # registry 에는 core_model 만 들어있으니 core_model 의 state_dict 로 로드.
        if isinstance(self.model_path, str) and self.model_path.startswith(("models:/", "runs:/")):
            import mlflow.pytorch
            print(f"  [MLflow] registry/run 에서 모델 로드: {self.model_path}")
            loaded_core = mlflow.pytorch.load_model(self.model_path, map_location=self.device)
            wrapped_model.core_model.load_state_dict(loaded_core.state_dict())
        else:
            state = torch.load(self.model_path, map_location=self.device)
            wrapped_model.load_state_dict(state)
        wrapped_model = wrapped_model.to(self.device)
        wrapped_model.eval()

        coset_lut = None
        if self.config.model.coset_mode and circuit is not None:
            from qec_sim.decoders.lut import build_detector_lut
            coset_lut = build_detector_lut(circuit)

        return NeuralDecoder(model=wrapped_model, coset_lut=coset_lut)

    def _open_mlflow_run(self, shots: int, model_dir: str):
        """`mlflow.enable=True` 이면 run 을 열고 params/tags 를 기록.
        반환: (mlflow module, run) 또는 (None, None)."""
        if not self.config.mlflow.enable:
            return None, None
        import dataclasses as dc
        import mlflow
        from qec_sim.trainer.callbacks import MLflowCallback

        cfg = self.config.mlflow
        # experiment_name 이 None 이면 per-code 컨벤션으로 자동.
        exp_name = cfg.experiment_name or (
            f"{self.config.code.name}_d{self.config.code.distance}"
        )

        mlflow.set_tracking_uri(cfg.tracking_uri)
        existing = mlflow.get_experiment_by_name(exp_name)
        if existing is None and cfg.artifact_location:
            mlflow.create_experiment(
                name=exp_name,
                artifact_location=cfg.artifact_location,
            )
        mlflow.set_experiment(experiment_name=exp_name)
        run = mlflow.start_run(run_name=cfg.run_name)
        print(f"  [MLflow] eval run started: experiment='{exp_name}' run_id={run.info.run_id}")

        params = MLflowCallback._flatten(dc.asdict(self.config))
        params["eval.shots"] = str(shots)
        params["eval.model_path"] = str(self.model_path) if self.model_path else ""
        items = list(params.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i:i + 100]))

        # source.name 도 ":eval" 접미사로 덮어써서 train 스크립트와 구분 (UI 의 source filter 영향).
        # mlflow.runType = "genai_evaluate" 박아서 사이드바 "Evaluation runs" 탭 분류 유도
        # (분류 결과 "Training runs" 에서 빠지길 기대 — 가설).
        tags = {
            "run_type": "eval",
            "decoder": self.config.decoder.name,
            "mlflow.source.name": "main.py:eval",
            "mlflow.runType": "genai_evaluate",
            **cfg.tags,
        }
        mlflow.set_tags(tags)

        if self.config_path and os.path.exists(self.config_path):
            mlflow.log_artifact(self.config_path, artifact_path="config")

        return mlflow, run

    @staticmethod
    def _parse_stim_filename(path: str) -> dict:
        """.stim 파일명에서 메타데이터를 best-effort 파싱 (BB 컨벤션 우선).

        모든 row가 동일 CSV 스키마를 갖도록 키(p/basis/nkd/d/file)는 항상 포함하고,
        매치 안 되면 None. file은 basename.
        """
        import os
        import re
        base = os.path.basename(path)
        def _search(pat, cast):
            m = re.search(pat, base)
            return cast(m.group(1)) if m else None
        return {
            "p": _search(r'p=([0-9.]+)', float),
            "basis": _search(r'c=(bivariate_bicycle_[XZ])', str),
            "nkd": _search(r'nkd=(\[\[[0-9,]+\]\])', str),
            "d": _search(r'(?:^|,)d=(\d+)', int),
            "file": base,
        }

    def _build_stim_file_sources(self):
        """code.stim_path glob을 나열해 파일당 (None, label, circuit) 소스를 만든다.

        sim은 None — sinter가 회로에서 직접 샘플링하므로 in-process 시뮬레이터 불필요.
        """
        import glob
        import stim
        path = self.config.code.stim_path
        if not path:
            raise ValueError("code.name='stim_file'은 code.stim_path 명시 필수.")
        matches = sorted(glob.glob(path))
        if not matches:
            raise FileNotFoundError(f"code.stim_path에 매치되는 파일 없음: {path}")
        print(f"  stim_file source: {len(matches)}개 파일 매치")
        sources = []
        for fp in matches:
            circuit = stim.Circuit.from_file(fp)
            label = self._parse_stim_filename(fp)
            sources.append((None, label, circuit))
        return sources

    def _collect_sinter(self, simulators_with_labels, decoder_name,
                        results, mlflow_mod):
        """sinter.collect로 algo decoder를 multiprocess 벤치. results/mlflow 채움."""
        import sinter

        ALGO = {"mwpm", "belief_matching", "belief_matching_fast", "bp_osd", "bp_lsd"}
        if decoder_name not in ALGO:
            raise ValueError(
                f"engine='sinter'는 algo decoder만 지원합니다 (받은: {decoder_name}). "
                f"지원: {sorted(ALGO)}. neural은 engine='inprocess'를 사용하세요."
            )

        mk = dict(self.config.decoder.model_kwargs)
        custom = {}
        builtin = []
        if decoder_name == "mwpm":
            task_dec = "pymatching"          # sinter 내장 (graphlike DEM 전용)
            builtin = ["pymatching"]
        elif decoder_name == "bp_osd":
            from qec_sim.decoders.sinter_bp_osd import BpOsdSinterDecoder
            task_dec = "bp_osd"
            custom["bp_osd"] = BpOsdSinterDecoder(**mk)
        elif decoder_name == "bp_lsd":
            # ldpc 내장 SinterLsdDecoder는 lsd_method 미노출 → 항상 LSD_0(order 0).
            # order>0(LSD_CS/LSD_E)을 쓰려면 자체 adapter 필요.
            from qec_sim.decoders.sinter_bp_lsd import BpLsdSinterDecoder
            task_dec = "bp_lsd"
            custom["bp_lsd"] = BpLsdSinterDecoder(**mk)
        elif decoder_name == "belief_matching_fast":
            from qec_sim.decoders.sinter_belief_matching_fast import BeliefMatchingFastSinterDecoder
            task_dec = "belief_matching_fast"
            custom["belief_matching_fast"] = BeliefMatchingFastSinterDecoder(**mk)
        elif decoder_name == "belief_matching":
            from beliefmatching import BeliefMatchingSinterDecoder
            task_dec = "belief_matching"
            custom["belief_matching"] = BeliefMatchingSinterDecoder(**mk)

        tasks = [
            sinter.Task(circuit=circuit, decoder=task_dec, json_metadata=label)
            for (_sim, label, circuit) in simulators_with_labels
        ]

        collect_kwargs = dict(
            num_workers=self.config.simulation.workers,
            tasks=tasks,
            custom_decoders=custom,
            max_shots=self.config.simulation.max_shots,
            max_errors=self.config.simulation.max_errors,
            print_progress=True,
        )
        if builtin:
            collect_kwargs["decoders"] = builtin

        print(f"  sinter collect: {len(tasks)} tasks × {task_dec}  "
              f"(workers={self.config.simulation.workers}, "
              f"max_shots={self.config.simulation.max_shots}, "
              f"max_errors={self.config.simulation.max_errors})", flush=True)

        # sinter 자체 진행표(print_progress, stderr)가 라이브 뷰를 제공. 우리 per-task
        # LER은 collect 종료 후 배치로 정리 출력 (progress_callback의 new_stats는 델타
        # 배치라 per-task 누적이 아니어서 스트리밍엔 부적합).
        stats = sinter.collect(**collect_kwargs)

        for idx, s in enumerate(stats):
            md = dict(s.json_metadata or {})
            ler = s.errors / s.shots if s.shots else float("nan")
            row = {**md, "shots": s.shots, "errors": s.errors,
                   "decoder": decoder_name, "ler": ler, "wall_seconds": s.seconds}
            results.append(row)
            label_str = ", ".join(f"{k}={v}" for k, v in md.items() if k != "file")
            print(f"{label_str} | LER: {ler:.4%} ({s.errors}/{s.shots})", flush=True)
            if mlflow_mod is not None:
                mlflow_mod.log_metric("ler", ler, step=idx)
                p = md.get("p")
                if isinstance(p, (int, float)) and not isinstance(p, bool):
                    mlflow_mod.log_metric("p", float(p), step=idx)

    def _run(self, shots: int, timestamp: str):
        decoder_name = self.config.decoder.name
        print(f"평가 시작 (Device: {self.device}, shots/noise: {shots:,})")
        print(f"디코더: {decoder_name}")
        if self.model_path:
            print(f"모델: {self.model_path}")
        print()

        # 1. 시뮬레이터 목록 구성 (backend에 따라 분기)
        from qec_sim.trainer.factory import _build_simulator_pool
        backend = self.config.simulation.backend
        results = []
        model_dir = self._resolve_output_dir(timestamp)
        mlflow_mod, mlflow_run = self._open_mlflow_run(shots, model_dir)

        if backend == 'stim':
            if self.config.code.name == 'stim_file':
                # 외부 .stim 회로 (BB/qLDPC 등). 노이즈 baked-in → noise sweep 없이
                # glob 매치 파일당 하나의 source. sinter 엔진 전용 (in-process 샘플러 없음).
                if self.config.simulation.engine != 'sinter':
                    raise ValueError(
                        "code.name='stim_file'은 simulation.engine='sinter' 전용입니다 "
                        "(파일 회로엔 in-process 샘플러 경로가 없음)."
                    )
                simulators_with_labels = self._build_stim_file_sources()
            else:
                noise_configs = self.config.get_expanded_noise_configs()
                simulators_with_labels = []
                for noise_cfg in noise_configs:
                    circuit = build_circuit(
                        self.config.code.name, self.config.code, noise_cfg
                    ).build()
                    sim = CircuitNoiseSimulator(circuit, noise_cfg)
                    label = {"p_gate": noise_cfg.p_gate, "p_meas": noise_cfg.p_meas,
                             "p_corr": noise_cfg.p_corr}
                    simulators_with_labels.append((sim, label, circuit))
        else:
            # pauli_plus: list-valued 필드를 Cartesian product로 확장
            from qec_sim.circuit.pauli_plus import PauliPlusSimulator
            from qec_sim.config.schema import NoiseParams
            noise_configs = self.config.get_expanded_pauli_plus_configs()
            if self.config.simulation.pauli_plus is None:
                raise KeyError("backend='pauli_plus'인 경우 simulation.pauli_plus 블록이 필요합니다.")
            pp_cfg_keys = list(self.config.simulation.pauli_plus.keys())
            simulators_with_labels = []
            for noise in noise_configs:
                sim = PauliPlusSimulator(self.config.code, noise)
                label = {k: getattr(noise, k) for k in pp_cfg_keys if hasattr(noise, k)}
                # MWPM의 DEM 또는 coset LUT 생성에 stim 회로가 필요.
                # leakage/crosstalk은 DEM에 표현 불가 — Pauli만 본다.
                needs_circuit = (
                    decoder_name in ("mwpm", "belief_matching", "belief_matching_fast", "bp_osd")
                    or self.config.model.coset_mode
                )
                proxy_circuit = None
                if needs_circuit:
                    proxy_circuit = build_circuit(
                        self.config.code.name,
                        self.config.code,
                        NoiseParams(p_gate=noise.p_2q, p_meas=noise.p_meas, p_corr=0.0),
                    ).build()
                simulators_with_labels.append((sim, label, proxy_circuit))

        # 2. neural_decoder는 모델을 미리 로드 (첫 circuit으로 LUT 생성)
        neural_decoder = None
        if decoder_name == "neural_decoder":
            first_circuit = simulators_with_labels[0][2] if simulators_with_labels else None
            neural_decoder = self._build_neural_decoder(circuit=first_circuit)

        print(f"{'backend':>12}: {backend}")
        print(f"{'LER':>12}")
        print("-" * 45)

        # sinter 엔진: multiprocess + adaptive stopping으로 한 번에 collect.
        # results/mlflow를 채운 뒤 in-process 루프는 건너뜀 (리스트를 비워 no-op).
        if self.config.simulation.engine == 'sinter':
            self._collect_sinter(simulators_with_labels, decoder_name,
                                 results, mlflow_mod)
            simulators_with_labels = []

        for idx, (sim, label, circuit) in enumerate(simulators_with_labels):
            raw = sim.generate_data(shots=shots)
            syndromes, observables = raw['syndromes'], raw['observables']

            if decoder_name == "neural_decoder":
                preds = neural_decoder.decode_batch(
                    syndromes, batch_size=4096,
                    soft_measurements=raw.get('soft_measurements'),
                )
            elif decoder_name == "mwpm":
                if circuit is None:
                    raise ValueError("mwpm decoder는 stim backend에서만 지원됩니다.")
                dem = circuit.detector_error_model(decompose_errors=True)
                mwpm = MWPMDecoder(error_model=dem)
                preds = mwpm.decode_batch(syndromes)
            elif decoder_name == "belief_matching":
                if circuit is None:
                    raise ValueError("belief_matching decoder는 stim backend에서만 지원됩니다.")
                dem = circuit.detector_error_model(decompose_errors=True)
                bm = BeliefMatchingDecoder(error_model=dem)
                preds = bm.decode_batch(syndromes)
            elif decoder_name == "belief_matching_fast":
                if circuit is None:
                    raise ValueError("belief_matching_fast decoder는 stim backend에서만 지원됩니다.")
                dem = circuit.detector_error_model(decompose_errors=True)
                bm_fast = BeliefMatchingFastDecoder(error_model=dem)
                preds = bm_fast.decode_batch(syndromes)
            elif decoder_name == "bp_osd":
                if circuit is None:
                    raise ValueError("bp_osd decoder는 stim backend에서만 지원됩니다.")
                dem = circuit.detector_error_model(decompose_errors=False)
                bp_osd = BpOsdDecoder(error_model=dem, **self.config.decoder.model_kwargs)
                preds = bp_osd.decode_batch(syndromes)
            else:
                raise ValueError(f"지원하지 않는 decoder: {decoder_name}")

            ler = float(np.mean(np.any(preds != observables, axis=1)))

            row = {**label, "shots": shots, "decoder": decoder_name, "ler": ler}
            results.append(row)
            label_str = ", ".join(f"{k}={v}" for k, v in label.items())
            print(f"{label_str} | LER: {ler:.4%}")

            if mlflow_mod is not None:
                # in-run curve view: step=idx 의 ler/노이즈 스칼라
                mlflow_mod.log_metric("ler", ler, step=idx)
                for k, v in label.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow_mod.log_metric(k, float(v), step=idx)
                # table-view 비교용 flat metric (단, key 에 점/특수문자 회피)
                tag = "_".join(f"{k}_{v}" for k, v in label.items()).replace('.', 'p')
                mlflow_mod.log_metric(f"ler_at__{tag}", ler)

        # 3. 저장
        self._save_results(results, model_dir, timestamp)

        # 4. mlflow artifact + run close
        if mlflow_mod is not None:
            csv_path = os.path.join(model_dir, f"eval_{timestamp}.csv")
            log_path = os.path.join(model_dir, f"eval_{timestamp}.log")
            if os.path.exists(csv_path):
                mlflow_mod.log_artifact(csv_path)
            if os.path.exists(log_path):
                mlflow_mod.log_artifact(log_path)
            mlflow_mod.end_run()

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
