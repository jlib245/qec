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
