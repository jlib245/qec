import os
import json
import random
import datetime
import traceback
import numpy as np
import torch

from qec_sim.config.schema import ExperimentConfig
from qec_sim.trainer.factory import ComponentFactory
from qec_sim.trainer.trainer import Trainer
from qec_sim.metrics.evaluator import Evaluator
from qec_sim.metrics.registry import build_criterion
from qec_sim.trainer.callbacks import (
    CSVLogger, RunLogger, ConfigSaver,
    BestModelSaver, Checkpoint, EarlyStopping, MLflowCallback,
)


class TrainingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ExperimentConfig.from_yaml(config_path)
        from qec_sim.core.interfaces import get_best_device
        self.device = get_best_device()
        self.workspace = {}
        self._phase = "init"

    def _setup_workspace(self):
        from qec_sim.trainer.utils import timestamped_output_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = timestamped_output_dir(self.config.training.output_dir, timestamp)
        os.makedirs(root, exist_ok=True)
        self.workspace = {
            "root":        root,
            "csv_log":     os.path.join(root, "training_log.csv"),
            "run_log":     os.path.join(root, "run.log"),
            "config":      os.path.join(root, "config.yaml"),
            "best_model":  os.path.join(root, "best_model.pth"),
            "checkpoint":  os.path.join(root, "checkpoint.pth"),
        }

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        self._setup_workspace()
        with RunLogger(log_path=self.workspace["run_log"]):
            try:
                self._run_inner()
            except BaseException as exc:
                self._dump_error(exc)
                raise

    def _dump_error(self, exc: BaseException):
        root = self.workspace["root"]
        tb_str = traceback.format_exc()
        log_path = os.path.join(root, "error.log")
        json_path = os.path.join(root, "error.json")

        with open(log_path, "w") as f:
            f.write(tb_str)

        record = {
            "timestamp":         datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "config_path":       self.config_path,
            "phase":             self._phase,
            "exception_type":    type(exc).__name__,
            "exception_module":  type(exc).__module__,
            "exception_message": str(exc),
            "device":            str(self.device),
            "output_dir":        root,
            "traceback":         tb_str,
        }
        with open(json_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"\n학습 실패 (phase={self._phase}). 저장: {log_path}, {json_path}")

    def _run_inner(self):
        self._phase = "setup"
        seed = self.config.training.seed
        if seed is not None:
            self._set_seed(seed)
            print(f"Seed 고정: {seed}")

        print(f"학습 파이프라인 시작 (Device: {self.device})")
        print(f"결과 저장 위치: {self.workspace['root']}\n")

        datamodule, wrapped_model = ComponentFactory.build_system(self.config)
        wrapped_model = wrapped_model.to(self.device)

        # Optional: load pretrained weights for fine-tuning.
        pretrained = self.config.model.pretrained
        if pretrained:
            state = torch.load(pretrained, map_location=self.device)
            if isinstance(state, dict) and "core_model" in state:
                state = state["core_model"]
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            # Try loading on wrapped_model first; fall back to core model.
            try:
                wrapped_model.load_state_dict(state, strict=False)
            except Exception:
                wrapped_model.core_model.load_state_dict(state, strict=False)
            print(f"  [Pretrained] 가중치 로드: {pretrained}")

        self._phase = "data_prepare"
        print("데이터를 준비합니다...")
        datamodule.strategy.prepare()
        train_loader, val_loader = datamodule.get_loaders()

        self._phase = "train_setup"
        criterion = build_criterion(
            self.config.training.criterion['name'],
            **self.config.training.criterion.get('kwargs', {})
        )
        evaluator = Evaluator(device=self.device, criterion=criterion)

        optimizer = getattr(torch.optim, self.config.training.optimizer['name'])(
            wrapped_model.parameters(),
            **self.config.training.optimizer['kwargs']
        )

        scheduler = None
        if self.config.training.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.config.training.scheduler['name'])(
                optimizer,
                **self.config.training.scheduler.get('kwargs', {})
            )

        callbacks = [
            ConfigSaver(src_path=self.config_path,        dst_path=self.workspace["config"]),
            CSVLogger(log_path=self.workspace["csv_log"]),
            BestModelSaver(save_path=self.workspace["best_model"], monitor='val_loss'),
            Checkpoint(save_path=self.workspace["checkpoint"]),
            EarlyStopping(patience=self.config.training.early_stopping['patience'], monitor='val_loss'),
        ]
        if self.config.mlflow.enable:
            callbacks.append(MLflowCallback(
                mlflow_cfg=self.config.mlflow,
                experiment_cfg=self.config,
                config_src_path=self.config_path,
                workspace=self.workspace,
            ))

        # online 모드: dataset이 epoch 크기를 직접 제어 → Trainer steps 제한 불필요
        is_online = self.config.training.data_mode == "online"
        trainer = Trainer(
            wrapped_model=wrapped_model,
            evaluator=evaluator,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            train_steps=None if is_online else self.config.training.train_steps,
            val_steps=None if is_online else self.config.training.val_steps,
        )

        self._phase = "train"
        trainer.fit(epochs=self.config.training.epochs)

        print(f"\n학습 완료. 저장 위치: {self.workspace['root']}")
