# qec_sim/trainer/callbacks.py
import os
import sys
import csv
import shutil
import torch


class Callback:
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs=None): pass
    def on_train_end(self, trainer): pass


# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────

class CSVLogger(Callback):
    """에포크별 지표를 CSV에 기록합니다."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history = []

    def on_epoch_end(self, trainer, epoch, logs=None):
        entry = dict(logs or {})
        entry['epoch'] = epoch + 1
        self.history.append(entry)
        os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)


class RunLogger:
    """stdout + stderr(tqdm 포함) 모두 파일에 기록 — 터미널에 보이는 그대로.

    Context manager. pipeline 진입 직후에 켜서 데이터 준비 등 모든 단계의 출력을 캡처.
    `\\r`는 현재 진행 중인 줄을 리셋하고 `\\n`는 그 줄을 commit하는 식으로
    터미널이 화면에 보여주는 결과를 그대로 파일에 반영. → IDE/cat 모두 한 줄짜리.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
        self._file = _ProgressAwareFile(self.log_path)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._file)
        sys.stderr = _Tee(self._orig_stderr, self._file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._file.close()
        return False


class _Tee:
    """write() 호출을 두 스트림에 동시에 전달합니다."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class _ProgressAwareFile:
    """`\\r`/`\\n`를 터미널 의미대로 처리하는 file-like wrapper.

    파일 내용 = (지금까지 commit된 줄들) + (현재 진행중인 부분 줄). `\\r`가 들어오면
    진행중 부분만 리셋하고, `\\n`가 들어오면 진행중 부분을 한 줄로 commit. 매 write마다
    파일 전체를 truncate-rewrite하므로 한 번에 큰 사이즈로 쓰지 말 것.
    """

    def __init__(self, path: str):
        self._path = path
        self._committed = ""
        self._partial = ""
        self._fh = open(path, 'w', encoding='utf-8', buffering=1)

    def write(self, data: str):
        if not data:
            return
        for ch in data:
            if ch == '\r':
                self._partial = ""
            elif ch == '\n':
                self._committed += self._partial + '\n'
                self._partial = ""
            else:
                self._partial += ch
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(self._committed + self._partial)
        self._fh.flush()

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()


# ──────────────────────────────────────────────
# 설정 저장
# ──────────────────────────────────────────────

class ConfigSaver(Callback):
    """사용한 config 파일을 output 디렉토리에 복사합니다."""

    def __init__(self, src_path: str, dst_path: str):
        self.src_path = src_path
        self.dst_path = dst_path

    def on_train_begin(self, trainer):
        os.makedirs(os.path.dirname(self.dst_path) or '.', exist_ok=True)
        shutil.copy(self.src_path, self.dst_path)


# ──────────────────────────────────────────────
# 체크포인트 / 조기종료
# ──────────────────────────────────────────────

class BestModelSaver(Callback):
    """monitor 기준 최적 모델 가중치만 저장합니다. (추론 / 평가용)"""

    def __init__(self, save_path: str, monitor: str = 'val_loss'):
        self.save_path = save_path
        self.monitor = monitor
        self.best_value = float('inf')
        self.best_weights = None

    def on_epoch_end(self, trainer, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is not None and current < self.best_value:
            self.best_value = current
            self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
            torch.save(self.best_weights, self.save_path)
            print(f"  [BestModel] {self.monitor} {current:.4f} → '{self.save_path}' 저장")

    def on_train_end(self, trainer):
        if self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            print(f"  [BestModel] 최적 모델(Loss: {self.best_value:.4f})로 복원 완료.")


class Checkpoint(Callback):
    """매 에포크 학습 상태 전체를 저장합니다. (학습 재개용)

    저장 내용: 모델 가중치 + optimizer + scheduler + epoch
    재개 방법: Checkpoint.load(path, model, optimizer, scheduler)
    """

    def __init__(self, save_path: str):
        self.save_path = save_path

    def on_epoch_end(self, trainer, epoch, logs=None):
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        state = {
            'epoch':          epoch + 1,
            'model':          trainer.model.state_dict(),
            'optimizer':      trainer.optimizer.state_dict(),
            'scheduler':      trainer.scheduler.state_dict() if trainer.scheduler else None,
            'logs':           logs or {},
        }
        torch.save(state, self.save_path)

    @staticmethod
    def load(path: str, model, optimizer=None, scheduler=None) -> int:
        """저장된 체크포인트를 불러옵니다. 재개할 epoch 번호를 반환합니다."""
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state['model'])
        if optimizer and state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
        if scheduler and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])
        return state['epoch']


class EarlyStopping(Callback):
    """patience 에포크 동안 개선이 없으면 학습을 중단합니다."""

    def __init__(self, patience: int, monitor: str = 'val_loss'):
        self.patience = patience
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, logs=None):
        if self.patience <= 0:
            return
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        if current < self.best_value:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n  [EarlyStopping] {self.patience} 에포크 동안 개선 없음 → 조기 종료")
                trainer.stop_training = True


# ──────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────

class MLflowCallback(Callback):
    """MLflow tracking 통합. enable=True 일 때만 pipeline 에서 callbacks 에 추가.

    on_train_begin   tracking_uri / experiment / start_run / log_params / set_tags / log config yaml
    on_epoch_end     logs dict 의 모든 스칼라 metric 을 log_metric(step=epoch)
    on_train_end     workspace 의 best_model.pth, checkpoint.pth, training_log.csv, run.log 을 artifact 로 등록 후 end_run

    callback 순서상 가장 마지막에 추가해야 BestModelSaver/Checkpoint 가
    on_train_end 까지 갱신한 파일들을 artifact 로 잡을 수 있다.
    """

    _MAX_PARAM_VAL_LEN = 500

    def __init__(self, mlflow_cfg, experiment_cfg, config_src_path, workspace):
        self.cfg = mlflow_cfg
        self.experiment_cfg = experiment_cfg
        self.config_src_path = config_src_path
        self.workspace = workspace
        self._active_run = None

    @staticmethod
    def _flatten(d, prefix="", out=None):
        """nested dict → flat dot-notation keys. list/scalar 는 str 화."""
        if out is None:
            out = {}
        if isinstance(d, dict):
            if not d:
                out[prefix] = "{}"
                return out
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                MLflowCallback._flatten(v, key, out)
        else:
            s = str(d)
            if len(s) > MLflowCallback._MAX_PARAM_VAL_LEN:
                s = s[: MLflowCallback._MAX_PARAM_VAL_LEN - 3] + "..."
            out[prefix[:250] if prefix else "value"] = s
        return out

    @staticmethod
    def _sha256_with_sidecar(path: str) -> str:
        """파일 SHA256. `<path>.sha256` sidecar 가 본체보다 최신이면 캐시 사용."""
        import hashlib
        sc = path + ".sha256"
        if os.path.exists(sc) and os.path.getmtime(sc) >= os.path.getmtime(path):
            return open(sc).read().strip()
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        try:
            with open(sc, 'w') as f:
                f.write(digest)
        except OSError:
            pass  # 권한/디스크 문제는 무시 — 다음 호출에 재계산.
        return digest

    def _log_dataset_ref(self, path: str, role: str):
        """offline npz path 를 log_input(MetaDataset) + set_tags + (UI sidebar 용)
        mlflow.genai.datasets entity 3채널에 등록.

        dataset name 은 yaml 의 `mlflow.tags.dataset.<role>.name` 으로 override 가능.
        없으면 `<basename>_<code.name>_d<distance>_r<rounds>` 식으로 자동.
        code/distance/rounds 는 tag 로도 박아서 dataset 별 filter 시 유용.
        """
        import mlflow
        from mlflow.data.meta_dataset import MetaDataset
        from mlflow.data.sources import LocalArtifactDatasetSource

        if not path or not os.path.exists(path):
            return
        digest = self._sha256_with_sidecar(path)
        code = self.experiment_cfg.code
        basename = os.path.basename(path).rsplit('.', 1)[0]

        # .npz 내용물 inspect — sample count + gen seed metadata
        import numpy as _np
        try:
            with _np.load(path, allow_pickle=False) as _d:
                if 'observables' in _d.files:
                    n_samples = int(len(_d['observables']))
                elif 'logical_outcomes' in _d.files:
                    n_samples = int(len(_d['logical_outcomes']))
                else:
                    n_samples = None
                gen_seed = int(_d['_meta_seed']) if '_meta_seed' in _d.files else None
        except Exception:
            n_samples, gen_seed = None, None

        # yaml 에서 override 했는지 확인 — `mlflow.tags.dataset.train.name` 형태
        # 기본 name: code/distance 는 experiment_name 에 이미 있으니 생략, rounds/samples/seed 만.
        user_tag_key = f"dataset.{role}.name"
        if user_tag_key in self.cfg.tags:
            name = self.cfg.tags[user_tag_key]
        else:
            parts = [basename, f"r{code.rounds}"]
            if n_samples is not None:
                parts.append(f"n{n_samples}")
            if gen_seed is not None:
                parts.append(f"s{gen_seed}")
            name = "_".join(parts)

        source = LocalArtifactDatasetSource(uri=path)
        ds = MetaDataset(source=source, name=name, digest=digest[:16])
        mlflow.log_input(ds, context=role)

        tag_map = {
            f"dataset.{role}.name":     name,
            f"dataset.{role}.path":     path,
            f"dataset.{role}.sha256":   digest,
            f"dataset.{role}.size":     str(os.path.getsize(path)),
            f"dataset.{role}.code":     code.name,
            f"dataset.{role}.distance": str(code.distance),
            f"dataset.{role}.rounds":   str(code.rounds),
        }
        if n_samples is not None:
            tag_map[f"dataset.{role}.n_samples"] = str(n_samples)
        if gen_seed is not None:
            tag_map[f"dataset.{role}.gen_seed"] = str(gen_seed)
        mlflow.set_tags(tag_map)

        # mlflow.genai.datasets entity 등록은 안 함 — GenAI eval 전용 API 라
        # 호출하면 train run 이 "GenAI dataset 에 대한 평가" 로 잘못 분류됨
        # (left sidebar 의 Evaluations > Evaluation runs 에 train 이 노출).
        # 사이드바 Datasets 탭 표시는 포기 — log_input + tag 채널 2개로 충분.

    def on_train_begin(self, trainer):
        import dataclasses as dc
        import mlflow

        # experiment_name 이 None 이면 per-code 컨벤션으로 자동.
        exp_name = self.cfg.experiment_name or (
            f"{self.experiment_cfg.code.name}_d{self.experiment_cfg.code.distance}"
        )

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        # set_experiment 은 artifact_location 인자가 없으니 신규 생성 시에만 적용.
        existing = mlflow.get_experiment_by_name(exp_name)
        if existing is None and self.cfg.artifact_location:
            mlflow.create_experiment(
                name=exp_name,
                artifact_location=self.cfg.artifact_location,
            )
        mlflow.set_experiment(experiment_name=exp_name)

        self._active_run = mlflow.start_run(run_name=self.cfg.run_name)
        run_id = self._active_run.info.run_id
        print(f"  [MLflow] run started: experiment='{exp_name}' run_id={run_id}")

        params = self._flatten(dc.asdict(self.experiment_cfg))
        # mlflow 는 한 번에 100개씩 묶어서 보내는 게 안전 (3.x 도 권장)
        items = list(params.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i:i + 100]))

        if self.cfg.tags:
            mlflow.set_tags(self.cfg.tags)

        # config yaml 은 on_train_end 에서 workspace 사본을 한 번만 푸시.
        # (on_train_begin 의 별도 subdir push 는 중복이라 제거)

        # offline 일 때만 dataset reference 등록 (online 은 시뮬레이터 generate).
        tc = self.experiment_cfg.training
        if tc.data_mode == "offline":
            self._log_dataset_ref(tc.train_path, "train")
            self._log_dataset_ref(tc.val_path,   "val")

    def on_epoch_end(self, trainer, epoch, logs=None):
        if self._active_run is None or not logs:
            return
        import mlflow
        for k, v in logs.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(k, float(v), step=epoch + 1)

    def on_train_end(self, trainer):
        if self._active_run is None:
            return
        import mlflow
        # 작은 파일만 mlflow artifact 로 푸시 — UI 편의용.
        # best_model.pth 는 registry (models:/m-...) 가 별도로 가지고 있으니 중복 회수.
        # checkpoint.pth 는 resume 용 로컬 전용 (mlflow 로 fetch 할 일 없음).
        for key in ("csv_log", "run_log", "config"):
            p = self.workspace.get(key)
            if p and os.path.exists(p):
                mlflow.log_artifact(p)

        # PyTorch model logging + (optional) registry 등록.
        # log_model 은 pickle 기반이라 custom class 사용 시 실패 가능 → try/except.
        if self.cfg.register_model:
            self._register_best_model(trainer)

        mlflow.end_run()
        self._active_run = None

    def _register_best_model(self, trainer):
        """Best weights 가 trainer.model 에 복원된 상태에서 호출.
        성공 시 registry 에 새 version + (옵션) alias 박음."""
        import mlflow
        run_id = self._active_run.info.run_id
        # 기본 등록명은 code-registry 이름(model.name) + distance.
        # yaml 의 model.name 이 곧 registry 의 클래스 키이므로 자연스럽게 매핑됨.
        name = self.cfg.registered_model_name or (
            f"{self.experiment_cfg.model.name}_d{self.experiment_cfg.code.distance}"
        )

        # 1) mlflow.pytorch.log_model 우선 시도 (load_model 한 줄로 복원 가능)
        model_uri = None
        try:
            import mlflow.pytorch
            # PreprocessorWrapper 가 아니라 내부 core_model 만 로그.
            # preprocessor 는 circuit/DEM 에 종속이라 직렬화 불안정.
            core = getattr(trainer.model, "core_model", trainer.model)
            mlflow.pytorch.log_model(pytorch_model=core, name="pytorch_model")
            model_uri = f"runs:/{run_id}/pytorch_model"
            print(f"  [MLflow] pytorch model logged: {model_uri}")
        except Exception as e:
            print(f"  [MLflow] pytorch.log_model 실패 ({type(e).__name__}: {e}). best_model.pth artifact 로 fallback.")
            best_path = self.workspace.get("best_model")
            if best_path and os.path.exists(best_path):
                model_uri = f"runs:/{run_id}/best_model.pth"

        if model_uri is None:
            print("  [MLflow] register 할 model artifact 없음 — registry 등록 스킵.")
            return

        # 2) registered model 생성 (없으면 자동 생성) + 새 version 등록
        try:
            mv = mlflow.register_model(model_uri=model_uri, name=name)
            print(f"  [MLflow] registry 등록: {name} version={mv.version}")
            # 3) alias 박기
            if self.cfg.register_alias:
                client = mlflow.MlflowClient()
                client.set_registered_model_alias(
                    name=name, alias=self.cfg.register_alias, version=mv.version,
                )
                print(f"  [MLflow] alias 박음: {name}@{self.cfg.register_alias} → v{mv.version}")
        except Exception as e:
            print(f"  [MLflow] registry 등록 실패 ({type(e).__name__}: {e})")
