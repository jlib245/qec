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
    BestModelSaver, Checkpoint, EarlyStopping,
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
            "last_model":  os.path.join(root, "last_model.pth"),
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
            RunLogger(log_path=self.workspace["run_log"]),
            CSVLogger(log_path=self.workspace["csv_log"]),
            BestModelSaver(save_path=self.workspace["best_model"], monitor='val_loss'),
            Checkpoint(save_path=self.workspace["checkpoint"]),
            EarlyStopping(patience=self.config.training.early_stopping['patience'], monitor='val_loss'),
        ]

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

        self._phase = "save"
        torch.save(wrapped_model.state_dict(), self.workspace["last_model"])
        print(f"\n학습 완료. 저장 위치: {self.workspace['root']}")
