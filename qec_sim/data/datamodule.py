# qec_sim/data/datamodule.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from typing import Callable, Optional, List, Tuple, Union


class CursorShuffleSampler(Sampler[int]):
    """전체 인덱스를 한 번 셔플한 뒤 epoch 경계와 무관하게 cursor가 이어진다.

    매 epoch당 `samples_per_epoch`개 인덱스를 yield하고, cursor가 끝에 도달하면
    새로 셔플한 다음 처음부터 다시 yield. dataset_size = N, samples_per_epoch = M
    이면 ⌈N/M⌉ epoch이 한 "전체 순회 사이클" — 그 동안은 같은 샘플이 두 번 안 나옴.
    """

    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: Optional[int] = None):
        self.N = dataset_size
        self.M = samples_per_epoch
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)
        self.perm = torch.randperm(self.N, generator=self.gen)
        self.cursor = 0

    def __iter__(self):
        end = self.cursor + self.M
        if end <= self.N:
            indices = self.perm[self.cursor:end]
            self.cursor = end
        else:
            tail = self.perm[self.cursor:]
            self.perm = torch.randperm(self.N, generator=self.gen)
            need = self.M - len(tail)
            indices = torch.cat([tail, self.perm[:need]])
            self.cursor = need
        yield from indices.tolist()

    def __len__(self):
        return self.M


def _worker_init_fn(worker_id: int):
    """DataLoader 워커마다 독립적이고 재현 가능한 seed를 설정합니다."""
    # 메인 프로세스 seed + worker_id → 워커별 고유 seed
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    import random
    random.seed(seed)


class _BitPackedTensor:
    """Stores a binary (N, F) tensor as packed uint8 with shape (N, ceil(F/8)).

    `__getitem__(idx)` (scalar or batch index) unpacks the requested row(s) on
    the fly and returns a torch.uint8 tensor of shape (..., F). Cache-friendly
    because the packed RAM footprint is 1/8 of the unpacked form.
    """

    def __init__(self, packed: np.ndarray, n_unpacked: int):
        # packed: shape (N, ceil(n_unpacked/8)), dtype uint8
        self.packed = np.ascontiguousarray(packed)
        self.n_unpacked = int(n_unpacked)

    def __len__(self):
        return self.packed.shape[0]

    @property
    def shape(self):
        return (self.packed.shape[0], self.n_unpacked)

    def __getitem__(self, idx):
        rows = self.packed[idx]                       # (..., ceil(F/8))
        bits = np.unpackbits(rows, axis=-1, count=self.n_unpacked)
        return torch.from_numpy(bits)


# Keys that are stored bit-packed when `_meta_bitpacked` is True.
# Each maps to the meta key holding the original (pre-pack) length.
_BITPACK_KEYS = {
    "syndromes":  "_meta_n_det",
    "edge_in_E0": "_meta_edge_E",
    "edge_in_E1": "_meta_edge_E",
}


def _maybe_unpack_full(arr_packed, n_unpacked, bitpacked):
    """Eagerly unpack a packed (N, ceil(F/8)) uint8 array to (N, F).

    Used for code paths (e.g. coset label computation) that need the full
    unpacked syndrome matrix up front rather than per-row.
    """
    if not bitpacked:
        return arr_packed
    return np.unpackbits(np.asarray(arr_packed), axis=-1, count=int(n_unpacked))


class QECRawDataset(Dataset):
    """전처리기가 명시한 데이터만 하드디스크에서 꺼내어 cpu_transform을 거쳐 반환합니다.

    coset_lut 가 주어지면 observables → coset label (class index) 로 변환.
    online path 의 OnlineQECDataset._yield_chunk_batches 와 대칭이라 동일 .npz 가
    coset / non-coset 양쪽에서 그대로 재사용 가능.

    Bit-packed cache 지원: .npz 의 `_meta_bitpacked` 가 True 이면 syndromes /
    edge_in_E0 / edge_in_E1 는 packed uint8 로 저장돼 있다. 그 키는 RAM 에 packed
    그대로 유지하고 `__getitem__` 에서 row 단위로 unpack 한다 (전체 unpack 은
    d=7 merged cache 의 경우 200GB+ 라 RAM 폭발).
    """

    def __init__(self, npz_path: str, required_keys: List[str],
                 cpu_transform: Optional[Callable] = None, coset_lut=None):
        data = np.load(npz_path)

        bitpacked = bool(data["_meta_bitpacked"].item()) \
            if "_meta_bitpacked" in data.files else False
        # Per-key unpacked lengths (only meaningful when bitpacked).
        unpacked_len = {}
        if bitpacked:
            for key, meta_key in _BITPACK_KEYS.items():
                if meta_key not in data.files:
                    raise ValueError(
                        f"bitpacked cache 인데 {meta_key} 가 없음 — "
                        f"build_routed_cache.py 가 메타를 안 적었음. file={npz_path}"
                    )
                unpacked_len[key] = int(data[meta_key].item())

        self.data_dict = {}
        for key in required_keys:
            if key not in data:
                raise ValueError(
                    f"전처리기가 '{key}'를 요구했으나 데이터셋에 없습니다. "
                    f"파일 내 키: {list(data.keys())}"
                )
            if bitpacked and key in _BITPACK_KEYS:
                self.data_dict[key] = _BitPackedTensor(
                    packed=np.asarray(data[key]),
                    n_unpacked=unpacked_len[key],
                )
            else:
                self.data_dict[key] = torch.tensor(data[key])

        # 'observables' 또는 'logical_outcomes' 키 모두 지원
        if 'observables' in data:
            obs = data['observables']
        elif 'logical_outcomes' in data:
            obs = data['logical_outcomes']
        else:
            raise ValueError(f"라벨 키('observables' 또는 'logical_outcomes')가 데이터셋에 없습니다. 파일 내 키: {list(data.keys())}")

        if coset_lut is not None:
            if 'syndromes' not in data:
                raise ValueError(
                    "coset_mode 사용 시 .npz 에 'syndromes' 키 필수 (coset label 계산용)."
                )
            from qec_sim.decoders.lut import compute_coset_labels
            # coset label 계산은 syndromes 전체를 한 번에 봐야 하므로 unpack.
            # (이 path 는 작은 cache 에서만 쓰는 게 안전 — full unpack 메모리 필요.)
            syn = _maybe_unpack_full(
                data['syndromes'],
                unpacked_len.get('syndromes'),
                bitpacked,
            )
            self.labels = torch.from_numpy(compute_coset_labels(syn, obs, coset_lut))
        else:
            self.labels = torch.tensor(obs)

        self.cpu_transform = cpu_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {key: self.data_dict[key][idx] for key in self.data_dict}
        if self.cpu_transform:
            sample = self.cpu_transform(sample)
        return sample, self.labels[idx]


class OfflineDataStrategy:
    """오프라인 .npz 파일에서 데이터를 로드하는 전략."""

    def __init__(self, config, required_keys: List[str],
                 cpu_transform: Optional[Callable] = None, coset_lut=None):
        self.config = config
        self.required_keys = required_keys
        self.cpu_transform = cpu_transform
        self.coset_lut = coset_lut
        self._train_dataset: Optional[QECRawDataset] = None
        self._val_dataset: Optional[QECRawDataset] = None

    def prepare(self) -> None:
        tc = self.config.training
        self._train_dataset = QECRawDataset(
            npz_path=tc.train_path,
            required_keys=self.required_keys,
            cpu_transform=self.cpu_transform,
            coset_lut=self.coset_lut,
        )
        self._val_dataset = QECRawDataset(
            npz_path=tc.val_path,
            required_keys=self.required_keys,
            cpu_transform=self.cpu_transform,
            coset_lut=self.coset_lut,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self._train_dataset is None or self._val_dataset is None:
            raise RuntimeError("prepare()를 먼저 호출하세요.")

        tc = self.config.training
        # num_workers=0 이면 prefetch_factor 사용 불가
        pf = tc.prefetch_factor if tc.num_workers > 0 else None

        # epoch_fraction 유효성은 TrainingConfig.__post_init__에서 이미 검증됨.
        N = len(self._train_dataset)
        samples_per_epoch = int(N * tc.epoch_fraction)
        train_sampler = CursorShuffleSampler(
            dataset_size=N,
            samples_per_epoch=samples_per_epoch,
            seed=tc.seed,
        )
        # persistent_workers=True: workers stay alive across epochs so we
        # avoid re-forking from a parent that may carry tens of GB of RSS
        # (cached datasets sit in RAM). Without this, fork can deadlock.
        persist = tc.num_workers > 0
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=tc.batch_size,
            sampler=train_sampler,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
            persistent_workers=persist,
        )
        val_loader = DataLoader(
            self._val_dataset,
            batch_size=tc.batch_size,
            shuffle=False,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
            persistent_workers=persist,
        )
        return train_loader, val_loader


class OnlineQECDataset(IterableDataset):
    """시뮬레이터에서 on-the-fly로 데이터를 생성하는 IterableDataset.

    chunk(시뮬 1회 호출분)를 받아 batch_size 단위 슬라이스로 yield.
    DataLoader는 batch_size=None으로 받아 collate 없이 그대로 통과 →
    sample-by-sample yield + collate 라운드트립 제거.
    """

    def __init__(
        self,
        simulator_pool,
        required_keys: List[str],
        epoch_samples: int,
        chunk_size: int,
        batch_size: int,
        cpu_transform: Optional[Callable] = None,
        cover_all_simulators: bool = False,
        coset_lut=None,
    ):
        self.simulator_pool = simulator_pool
        self.required_keys = required_keys
        self.epoch_samples = epoch_samples
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cpu_transform = cpu_transform
        self.cover_all_simulators = cover_all_simulators
        self.coset_lut = coset_lut

    def _yield_chunk_batches(self, raw):
        """chunk를 batch_size 단위로 슬라이스해 (batch_dict, batch_labels) yield."""
        n = len(raw['observables'])

        # zero-copy view — trainer가 .float()로 변환하므로 여기서는 dtype 유지.
        tensors = {
            k: torch.from_numpy(raw[k])
            for k in self.required_keys
            if k in raw
        }

        if self.coset_lut is not None:
            from qec_sim.decoders.lut import compute_coset_labels
            label_arr = torch.from_numpy(
                compute_coset_labels(raw['syndromes'], raw['observables'], self.coset_lut)
            )
        else:
            label_arr = torch.from_numpy(raw['observables'])

        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = {k: t[start:end] for k, t in tensors.items()}
            labels = label_arr[start:end]
            if self.cpu_transform:
                batch = self.cpu_transform(batch)
            # Optional filter: preprocessor injects `_keep_mask` (e.g. gap-routed
            # fine-tuning keeps only gap < threshold). Apply to batch + labels.
            if "_keep_mask" in batch:
                mask = batch.pop("_keep_mask")
                if mask.sum() == 0:
                    continue
                batch = {k: v[mask] for k, v in batch.items()}
                labels = labels[mask]
            yield batch, labels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # fork된 워커들은 simulator.rng 상태를 공유하므로 워커별 reseed 필수.
            # PyTorch가 epoch마다 새 base_seed를 뽑으므로 epoch 간 다양성도 확보됨.
            base_seed = torch.initial_seed() % (2 ** 32)
            ss = np.random.SeedSequence([base_seed, worker_id])
            sims = self.simulator_pool.get_all_simulators()
            for sim, child in zip(sims, ss.spawn(len(sims))):
                sim.reseed(child)

        if self.cover_all_simulators:
            yield from self._iter_all_simulators(num_workers, worker_id)
        else:
            yield from self._iter_random(num_workers, worker_id)

    @staticmethod
    def _worker_samples(total, num_workers, worker_id):
        """워커별 샘플 수 계산. 나머지는 마지막 워커에게."""
        per_worker = total // num_workers
        if worker_id == num_workers - 1:
            per_worker += total - per_worker * num_workers
        return per_worker

    def _iter_all_simulators(self, num_workers, worker_id):
        """모든 시뮬레이터를 균등하게 순회하며 데이터 생성."""
        all_sims = self.simulator_pool.get_all_simulators()
        per_sim = self.epoch_samples // len(all_sims)

        for sim in all_sims:
            per_worker = self._worker_samples(per_sim, num_workers, worker_id)
            generated = 0
            while generated < per_worker:
                n_get = min(self.chunk_size, per_worker - generated)
                raw = sim.generate_data(n_get)
                yield from self._yield_chunk_batches(raw)
                generated += n_get

    def _iter_random(self, num_workers, worker_id):
        """랜덤 시뮬레이터로 데이터 생성 (기존 동작)."""
        per_worker = self._worker_samples(self.epoch_samples, num_workers, worker_id)
        generated = 0
        while generated < per_worker:
            sim = self.simulator_pool.get_random_simulator()
            n_get = min(self.chunk_size, per_worker - generated)
            raw = sim.generate_data(n_get)
            yield from self._yield_chunk_batches(raw)
            generated += n_get


class OnlineDataStrategy:
    """시뮬레이터에서 실시간으로 데이터를 생성하는 전략."""

    def __init__(self, config, simulator_pool, required_keys: List[str],
                 cpu_transform: Optional[Callable] = None, coset_lut=None):
        self.config = config
        self.simulator_pool = simulator_pool
        self.required_keys = required_keys
        self.cpu_transform = cpu_transform
        self.coset_lut = coset_lut
        self._train_dataset: Optional[OnlineQECDataset] = None
        self._val_dataset: Optional[OnlineQECDataset] = None

    def prepare(self) -> None:
        tc = self.config.training
        # train_steps/val_steps 유효성은 TrainingConfig.__post_init__에서 검증.
        train_samples = tc.train_steps
        val_samples = tc.val_steps

        # val_steps를 시뮬레이터 수의 배수로 올림 (균등 분배 보장)
        num_sims = len(self.simulator_pool.get_all_simulators())
        if val_samples % num_sims != 0:
            val_samples = ((val_samples // num_sims) + 1) * num_sims

        self._train_dataset = OnlineQECDataset(
            simulator_pool=self.simulator_pool,
            required_keys=self.required_keys,
            epoch_samples=train_samples,
            chunk_size=tc.chunk_size,
            batch_size=tc.batch_size,
            cpu_transform=self.cpu_transform,
            coset_lut=self.coset_lut,
        )
        self._val_dataset = OnlineQECDataset(
            simulator_pool=self.simulator_pool,
            required_keys=self.required_keys,
            epoch_samples=val_samples,
            chunk_size=tc.chunk_size,
            batch_size=tc.batch_size,
            cpu_transform=self.cpu_transform,
            cover_all_simulators=True,
            coset_lut=self.coset_lut,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self._train_dataset is None or self._val_dataset is None:
            raise RuntimeError("prepare()를 먼저 호출하세요.")

        tc = self.config.training
        pf = tc.prefetch_factor if tc.num_workers > 0 else None

        # batch_size=None: dataset이 이미 batch 단위로 yield하므로 collate 안 함.
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=None,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            self._val_dataset,
            batch_size=None,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        return train_loader, val_loader


class QECDataModule:
    """데이터 전략을 감싸는 모듈."""

    def __init__(self, strategy: Union[OfflineDataStrategy, OnlineDataStrategy]):
        self.strategy = strategy

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.strategy.get_loaders()
