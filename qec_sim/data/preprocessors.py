# qec_sim/data/preprocessors.py
import torch
from collections import defaultdict
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor


@register_preprocessor("spatial_grid")
class SpatialGridPreprocessor(BasePreprocessor):
    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(
            detector_coords=circuit.get_detector_coordinates(),
            num_detectors=circuit.num_detectors,
        )

    def __init__(self, detector_coords: dict, num_detectors: int, **kwargs):
        super().__init__()
        self.num_detectors = num_detectors

        self._required_keys = ["syndromes"]

        all_x = [c[0] for c in detector_coords.values()]
        all_y = [c[1] for c in detector_coords.values()]
        all_t = [c[2] for c in detector_coords.values()] if len(list(detector_coords.values())[0]) > 2 else [0]

        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) // 2.0) + 1
        self.grid_h = int((max(all_y) - min_y) // 2.0) + 1

        unique_t = sorted(list(set(all_t)))
        self.input_depth = len(unique_t)
        self.out_channels = self.input_depth

        det_indices, c_indices, h_indices, w_indices = [], [], [], []
        t_map = {t: i for i, t in enumerate(unique_t)}
        for det_idx in range(num_detectors):
            if det_idx in detector_coords:
                coords = detector_coords[det_idx]
                x, y = coords[0], coords[1]
                t = coords[2] if len(coords) > 2 else 0
                det_indices.append(det_idx)
                c_indices.append(t_map[t])
                h_indices.append(int((y - min_y) // 2.0))
                w_indices.append(int((x - min_x) // 2.0))

        # persistent=False: state_dict 제외 + wrapper.to(device) 시 자동 이동.
        self.register_buffer('det_idx', torch.tensor(det_indices, dtype=torch.long), persistent=False)
        self.register_buffer('c_idx',   torch.tensor(c_indices,   dtype=torch.long), persistent=False)
        self.register_buffer('h_idx',   torch.tensor(h_indices,   dtype=torch.long), persistent=False)
        self.register_buffer('w_idx',   torch.tensor(w_indices,   dtype=torch.long), persistent=False)

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"in_channels": self.out_channels, "grid_h": self.grid_h, "grid_w": self.grid_w}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_syn = batch_data["syndromes"]
        batch_size = batch_syn.size(0)
        device = batch_syn.device

        grid = torch.full((batch_size, self.out_channels, self.grid_h, self.grid_w), -0.5, device=device)
        grid[:, self.c_idx, self.h_idx, self.w_idx] = batch_syn[:, self.det_idx]

        return grid


@register_preprocessor("soft_grid")
class SoftGridPreprocessor(BasePreprocessor):
    """
    IQ soft measurement 기반 spatial grid 전처리기.
    soft_measurements (shots, rounds, n_ancilla) → (shots, rounds, H, W) float tensor.

    각 채널은 한 라운드의 soft detection event:
      - 채널 0: soft_meas[0]  (첫 라운드 posterior, 완벽한 |0⟩ 참조 대비)
      - 채널 r>0: soft_xor(soft_meas[r], soft_meas[r-1])  (연속 라운드 XOR)
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        if simulator_pool is None:
            raise ValueError("SoftGridPreprocessor requires simulator_pool (pauli_plus backend)")
        sim = simulator_pool.get_random_simulator()
        return cls(
            ancilla_coords=sim.get_ancilla_coordinates(),
            rounds=config.code.rounds,
        )

    def __init__(self, ancilla_coords: dict, rounds: int, **kwargs):
        super().__init__()
        self.rounds = rounds
        n_ancilla = len(ancilla_coords)

        all_x = [c[0] for c in ancilla_coords.values()]
        all_y = [c[1] for c in ancilla_coords.values()]
        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) / 2) + 1
        self.grid_h = int((max(all_y) - min_y) / 2) + 1

        h_indices = [int((ancilla_coords[i][1] - min_y) / 2) for i in range(n_ancilla)]
        w_indices = [int((ancilla_coords[i][0] - min_x) / 2) for i in range(n_ancilla)]

        self.register_buffer('h_idx', torch.tensor(h_indices, dtype=torch.long), persistent=False)
        self.register_buffer('w_idx', torch.tensor(w_indices, dtype=torch.long), persistent=False)

    @property
    def required_data_keys(self) -> List[str]:
        return ["soft_measurements"]

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"in_channels": self.rounds, "grid_h": self.grid_h, "grid_w": self.grid_w}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # soft_meas: (batch, rounds, n_ancilla)
        soft_meas = batch_data["soft_measurements"]
        batch_size = soft_meas.size(0)
        device = soft_meas.device

        # soft detection events: round 0 그대로, round 1+ 는 soft XOR
        soft_r0 = soft_meas[:, 0:1, :]                                          # (B, 1, A)
        m_curr = soft_meas[:, 1:, :]
        m_prev = soft_meas[:, :-1, :]
        soft_rest = m_curr + m_prev - 2 * m_curr * m_prev                       # (B, R-1, A)
        soft_det = torch.cat([soft_r0, soft_rest], dim=1)                        # (B, R, A)

        grid = torch.zeros(batch_size, self.rounds, self.grid_h, self.grid_w, device=device)
        grid[:, :, self.h_idx, self.w_idx] = soft_det

        return grid


@register_preprocessor("flat")
class FlatPreprocessor(BasePreprocessor):
    """MLP 계열 모델용 flat 전처리기. syndromes을 1D 텐서로 반환합니다."""

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(num_detectors=circuit.num_detectors)

    def __init__(self, num_detectors: int, **kwargs):
        super().__init__()
        self.num_detectors = num_detectors
        self._required_keys = ["syndromes"]

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"input_dim": self.num_detectors}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch_data["syndromes"].float()


def _extract_qubit_coords(circuit) -> Dict[int, tuple]:
    """Stim 회로에서 QUBIT_COORDS instruction을 모아 {qubit_idx: (x, y, ...)} 반환."""
    out = {}
    for instr in circuit.flattened():
        if instr.name != "QUBIT_COORDS":
            continue
        coords = tuple(instr.gate_args_copy())
        for tgt in instr.targets_copy():
            out[tgt.qubit_value] = coords
    return out


def _classify_rotated_surface_qubits(qubit_coords):
    """Rotated surface code 좌표 규칙에 따라 data/Z-anc/X-anc 분류.
       data: (x, y) 모두 odd. Z-anc: (x-y) % 4 == 0. X-anc: (x-y) % 4 == 2.
    """
    data, z_anc, x_anc = {}, {}, {}
    for q, (x, y) in qubit_coords.items():
        ix, iy = int(x), int(y)
        if ix % 2 == 1 and iy % 2 == 1:
            data[q] = (ix, iy)
        elif (ix - iy) % 4 == 0:
            z_anc[q] = (ix, iy)
        elif (ix - iy) % 4 == 2:
            x_anc[q] = (ix, iy)
        else:
            raise RuntimeError(f"unexpected qubit {q} at ({x},{y})")
    return data, z_anc, x_anc


@register_preprocessor("qct_qubit_centric")
class QCTPreprocessor(BasePreprocessor):
    """QCT (Qubit-centric Transformer)용 전처리기.

    Stim 회로의 detector_coords + qubit_coords로부터 다음을 계산해 모델에 주입:
      - n: 데이터 큐빗 수
      - m_Z, m_X: Z/X-stab ancilla 수
      - T_Z, T_X: Z/X-stab이 detector를 갖는 round 수
      - K_Z, K_X: 큐빗당 인접 Z/X-stab 최대 개수 (rotated SC에선 ≤2)
      - nbr_Z (n, K_Z): 데이터 큐빗 i의 인접 Z-stab ancilla idx (-1 = 패딩)
      - nbr_X (n, K_X)
      - det_Z_at (m_Z, T_Z): Z-stab a, round t의 detector idx
      - det_X_at (m_X, T_X)
      - mask (n, n): N(i) ∩ N(j) ≠ ∅ → True (unmasked)

    `gpu_transform`은 syndromes (B, num_detectors)를 그대로 float로 반환 — gather와
    sign 변환은 모델 forward에서 수행 (학습 가능한 W_e와 결합되므로).
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        self.num_detectors = circuit.num_detectors
        self._required_keys = ["syndromes"]
        self._build_topology(circuit)

    def _build_topology(self, circuit):
        qc = _extract_qubit_coords(circuit)
        data_q, z_anc_q, x_anc_q = _classify_rotated_surface_qubits(qc)

        # row-major (y, x) 정렬 — 결정론적 인덱싱
        data_sorted = sorted(data_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))
        z_sorted = sorted(z_anc_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))
        x_sorted = sorted(x_anc_q.items(), key=lambda kv: (kv[1][1], kv[1][0]))

        n = len(data_sorted)
        m_Z = len(z_sorted)
        m_X = len(x_sorted)

        z_coord_to_idx = {coord: i for i, (_, coord) in enumerate(z_sorted)}
        x_coord_to_idx = {coord: i for i, (_, coord) in enumerate(x_sorted)}

        nbr_Z, nbr_X = [[] for _ in range(n)], [[] for _ in range(n)]
        for di, (_, (x, y)) in enumerate(data_sorted):
            for dx, dy in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
                c = (x + dx, y + dy)
                if c in z_coord_to_idx:
                    nbr_Z[di].append(z_coord_to_idx[c])
                elif c in x_coord_to_idx:
                    nbr_X[di].append(x_coord_to_idx[c])

        K_Z = max(len(v) for v in nbr_Z)
        K_X = max(len(v) for v in nbr_X)
        nbr_Z_t = torch.full((n, K_Z), -1, dtype=torch.long)
        nbr_X_t = torch.full((n, K_X), -1, dtype=torch.long)
        for i in range(n):
            if nbr_Z[i]:
                nbr_Z_t[i, :len(nbr_Z[i])] = torch.tensor(nbr_Z[i], dtype=torch.long)
            if nbr_X[i]:
                nbr_X_t[i, :len(nbr_X[i])] = torch.tensor(nbr_X[i], dtype=torch.long)

        det_coords = circuit.get_detector_coordinates()
        z_det_per_round = defaultdict(dict)
        x_det_per_round = defaultdict(dict)
        for det_id, (cx, cy, ct) in det_coords.items():
            coord = (int(cx), int(cy))
            if coord in z_coord_to_idx:
                z_det_per_round[ct][z_coord_to_idx[coord]] = det_id
            elif coord in x_coord_to_idx:
                x_det_per_round[ct][x_coord_to_idx[coord]] = det_id
            else:
                raise RuntimeError(f"detector {det_id} at ({cx},{cy}) matches no ancilla")

        z_rounds = sorted(z_det_per_round.keys())
        x_rounds = sorted(x_det_per_round.keys())
        T_Z = len(z_rounds)
        T_X = len(x_rounds)

        det_Z_at = torch.zeros((m_Z, T_Z), dtype=torch.long)
        det_X_at = torch.zeros((m_X, T_X), dtype=torch.long)
        for ti, t in enumerate(z_rounds):
            for ai in range(m_Z):
                det_Z_at[ai, ti] = z_det_per_round[t][ai]
        for ti, t in enumerate(x_rounds):
            for ai in range(m_X):
                det_X_at[ai, ti] = x_det_per_round[t][ai]

        # Structure-aware mask (rotated SC: 코너 4, 엣지 6, 인테리어 9)
        mask = torch.zeros((n, n), dtype=torch.bool)
        nbrs_per_qubit = []
        for i in range(n):
            s = set()
            for a in nbr_Z[i]:
                s.add(("Z", a))
            for a in nbr_X[i]:
                s.add(("X", a))
            nbrs_per_qubit.append(s)
        for i in range(n):
            for j in range(n):
                if nbrs_per_qubit[i] & nbrs_per_qubit[j]:
                    mask[i, j] = True

        # 버퍼 등록 — wrapper.to(device) 시 자동 이동
        self.register_buffer("nbr_Z", nbr_Z_t, persistent=False)
        self.register_buffer("nbr_X", nbr_X_t, persistent=False)
        self.register_buffer("det_Z_at", det_Z_at, persistent=False)
        self.register_buffer("det_X_at", det_X_at, persistent=False)
        self.register_buffer("attn_mask", mask, persistent=False)

        self._meta = dict(n=n, m_Z=m_Z, m_X=m_X, T_Z=T_Z, T_X=T_X, K_Z=K_Z, K_X=K_X)

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        # 토폴로지 메타 + 버퍼 텐서를 모델 __init__에 전달
        return dict(
            **self._meta,
            nbr_Z=self.nbr_Z,
            nbr_X=self.nbr_X,
            det_Z_at=self.det_Z_at,
            det_X_at=self.det_X_at,
            attn_mask=self.attn_mask,
        )

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch_data["syndromes"].float()


# ─────────────────────────────────────────────────────────────────────────
# GapRoutedCNNPreprocessor
# ─────────────────────────────────────────────────────────────────────────
# Per-sample MWPM (standard + 2 augmented) → 4-channel grid + 6 scalars.
# Designed for the GapRoutedCNN model (qec_sim.models.gap_routed_cnn).
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("gap_routed_cnn")
class GapRoutedCNNPreprocessor(BasePreprocessor):
    """Compute MWPM features per syndrome and pack into a flat tensor.

    Grid channels (per-detector → placed on (T, H, W) grid):
      0  syndrome (detection events)
      1  edge_diff_mask           — endpoints of (E_0 △ E_1)
      2  w0_mask                  — per-detector E_0 edge weight sum
      3  w1_mask                  — per-detector E_1 edge weight sum

    Scalar features:
      gap, w_0, w_1, weight_sum, n_matched_norm, mwpm_pred
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        import numpy as np
        import pymatching
        from beliefmatching import detector_error_model_to_check_matrices
        from scipy import sparse

        if circuit is None:
            raise ValueError("GapRoutedCNNPreprocessor requires the stim circuit.")
        self._required_keys = ["syndromes"]
        self.num_detectors = circuit.num_detectors

        # MWPM (augmented only) — built from the DEM of this circuit.
        # Standard MWPM call is omitted: argmin(w_0, w_1) reproduces mwpm_pred
        # and min(w_0, w_1) reproduces its weight sum bit-identically.
        # For multi-p training the SimulatorPool samples mixed noise but we use
        # a single MWPM (built from this circuit's DEM, typically the median p).
        dem = circuit.detector_error_model(decompose_errors=True)
        mtx = detector_error_model_to_check_matrices(dem)
        priors = np.asarray(mtx.hyperedge_to_edge_matrix @ mtx.priors)
        w_static = -np.log(np.clip(priors, 1e-14, 1 - 1e-14)).astype(np.float64)
        self._edge_weights = w_static

        EC = sparse.csc_matrix(mtx.edge_check_matrix)
        EO = sparse.csr_matrix(mtx.edge_observables_matrix)
        obs_row = sparse.csr_matrix(EO.toarray()[0:1])
        EC_aug = sparse.vstack([EC, obs_row]).tocsc()
        self._M_aug = pymatching.Matching.from_check_matrix(
            EC_aug, weights=w_static,
            faults_matrix=EO,
            use_virtual_boundary_node=True,
        )
        self._edge_lookup_aug = self._build_edge_lookup(EC_aug)

        # detector grid coordinates
        raw = circuit.get_detector_coordinates()
        coords = np.array([raw[i] for i in range(self.num_detectors)])
        xs = sorted(set(coords[:, 0]));  x2i = {v: i for i, v in enumerate(xs)}
        ys = sorted(set(coords[:, 1]));  y2i = {v: i for i, v in enumerate(ys)}
        ts = sorted(set(coords[:, 2]));  t2i = {v: i for i, v in enumerate(ts)}
        det_idx = np.array([
            [t2i[coords[i, 2]], y2i[coords[i, 1]], x2i[coords[i, 0]]]
            for i in range(self.num_detectors)
        ], dtype=np.int64)
        self.T, self.H, self.W = len(ts), len(ys), len(xs)
        self.n_grid_channels = 4
        self.n_scalars = 6
        self.grid_dim = self.n_grid_channels * self.T * self.H * self.W

        # flat index per detector for scatter
        flat_idx = (det_idx[:, 0] * self.H * self.W
                    + det_idx[:, 1] * self.W
                    + det_idx[:, 2])
        self.register_buffer(
            "_flat_idx", torch.from_numpy(flat_idx).long(), persistent=False
        )

    # ----- helpers -----
    @staticmethod
    def _build_edge_lookup(EC):
        from scipy import sparse
        if not sparse.issparse(EC):
            EC = sparse.csc_matrix(EC)
        EC = EC.tocsc()
        n_edge = EC.shape[1]
        lookup = {}
        for e in range(n_edge):
            rows = EC.getcol(e).nonzero()[0]
            if rows.size == 1:
                lookup[(int(rows[0]), -1)] = e
                lookup[(-1, int(rows[0]))] = e
            elif rows.size == 2:
                a, b = int(rows[0]), int(rows[1])
                lookup[(a, b)] = e
                lookup[(b, a)] = e
        return lookup

    # ----- interface -----
    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": self.n_grid_channels,
                "n_scalars": self.n_scalars}

    def cpu_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Batched MWPM feature extraction.

        batch["syndromes"]: tensor or array (B, n_det). Per-shot 2 augmented
        MWPM runs (syndrome appended with 0/1); returns 4 per-detector channel
        batches + 6 scalars. mwpm_pred / weight_sum derived from argmin(w_0,w_1).
        """
        import numpy as np
        syn = batch["syndromes"]
        if isinstance(syn, torch.Tensor):
            syn_np = syn.cpu().numpy()
        else:
            syn_np = np.asarray(syn)
        syn_np = syn_np.astype(np.uint8)
        if syn_np.ndim == 1:
            syn_np = syn_np[None, :]
        B, n_det = syn_np.shape

        ch_syn  = syn_np.astype(np.float32)
        ch_diff = np.zeros((B, n_det), dtype=np.float32)
        ch_w0   = np.zeros((B, n_det), dtype=np.float32)
        ch_w1   = np.zeros((B, n_det), dtype=np.float32)
        scalars = np.zeros((B, 6), dtype=np.float32)

        for i in range(B):
            s = syn_np[i]
            syn0 = np.concatenate([s, [0]]).astype(np.uint8)
            syn1 = np.concatenate([s, [1]]).astype(np.uint8)
            e0 = self._M_aug.decode_to_edges_array(syn0)
            e1 = self._M_aug.decode_to_edges_array(syn1)

            set0, set1 = set(), set()
            w0 = 0.0
            for a, b in e0:
                a, b = int(a), int(b)
                key = (a, b) if a <= b else (b, a)
                set0.add(key)
                eid = self._edge_lookup_aug.get((a, b))
                w = float(self._edge_weights[eid]) if eid is not None else 0.0
                w0 += w
                if 0 <= a < n_det: ch_w0[i, a] += w
                if 0 <= b < n_det: ch_w0[i, b] += w
            w1 = 0.0
            for a, b in e1:
                a, b = int(a), int(b)
                key = (a, b) if a <= b else (b, a)
                set1.add(key)
                eid = self._edge_lookup_aug.get((a, b))
                w = float(self._edge_weights[eid]) if eid is not None else 0.0
                w1 += w
                if 0 <= a < n_det: ch_w1[i, a] += w
                if 0 <= b < n_det: ch_w1[i, b] += w
            for (a, b) in (set0 ^ set1):
                if 0 <= a < n_det: ch_diff[i, a] = 1.0
                if 0 <= b < n_det: ch_diff[i, b] = 1.0

            gap = abs(w0 - w1)
            if w0 <= w1:
                mwpm_pred, weight_sum, n_matched = 0, w0, len(e0)
            else:
                mwpm_pred, weight_sum, n_matched = 1, w1, len(e1)
            scalars[i] = [gap, w0, w1, weight_sum,
                          n_matched / max(1, n_det),
                          float(mwpm_pred)]

        return {
            "ch_syn":  torch.from_numpy(ch_syn),
            "ch_diff": torch.from_numpy(ch_diff),
            "ch_w0":   torch.from_numpy(ch_w0),
            "ch_w1":   torch.from_numpy(ch_w1),
            "scalars": torch.from_numpy(scalars),
        }

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack per-detector channels onto (B, 4, T, H, W) grid, concat with scalars."""
        ch_syn  = batch_data["ch_syn"]
        ch_diff = batch_data["ch_diff"]
        ch_w0   = batch_data["ch_w0"]
        ch_w1   = batch_data["ch_w1"]
        scalars = batch_data["scalars"]
        B = ch_syn.size(0)
        device = ch_syn.device

        # scatter channels onto flat (T*H*W) grid
        grid = torch.zeros(B, self.n_grid_channels, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        grid[:, 1, flat_idx] = ch_diff.float()
        grid[:, 2, flat_idx] = ch_w0.float()
        grid[:, 3, flat_idx] = ch_w1.float()

        return torch.cat([grid.reshape(B, -1), scalars.float()], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# GapRoutedCNNFilteredPreprocessor — gap-filtered training (fine-tune on
# routed sub-distribution, gap < threshold)
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("gap_routed_cnn_filtered")
class GapRoutedCNNFilteredPreprocessor(GapRoutedCNNPreprocessor):
    """Same as GapRoutedCNNPreprocessor but drops samples with gap >= threshold.

    Used for fine-tuning the CNN on the routed sub-distribution only
    (matches deployment-time CNN input). Inserts `_keep_mask` into the
    cpu_transform output; the dataset applies the mask to features + labels.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, gap_threshold: float = 2.0, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.gap_threshold = float(gap_threshold)

    def cpu_transform(self, batch):
        out = super().cpu_transform(batch)
        gaps = out["scalars"][:, 0]
        out["_keep_mask"] = (gaps < self.gap_threshold)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Syndrome3DPreprocessor — generic 3D syndrome grid (1 channel, no scalars)
# ─────────────────────────────────────────────────────────────────────────
# Detector coordinates from the circuit are arranged into a (T, H, W) grid.
# Only the syndrome (detection events) is placed; no DEM-derived per-shot
# features. Useful as a baseline for 3D CNN syndrome decoders.
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("syndrome_3d")
class Syndrome3DPreprocessor(BasePreprocessor):
    """Generic 3D syndrome grid preprocessor (1 grid channel, 0 scalars).

    Same (T, H, W) detector layout as GapRoutedCNNPreprocessor — usable as a
    baseline for any 3D CNN model that consumes the syndrome volume directly.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(circuit=circuit)

    def __init__(self, circuit, **kwargs):
        super().__init__()
        import numpy as np
        if circuit is None:
            raise ValueError("Syndrome3DPreprocessor requires the stim circuit.")
        self._required_keys = ["syndromes"]
        self.num_detectors = circuit.num_detectors

        raw = circuit.get_detector_coordinates()
        coords = np.array([raw[i] for i in range(self.num_detectors)])
        xs = sorted(set(coords[:, 0]));  x2i = {v: i for i, v in enumerate(xs)}
        ys = sorted(set(coords[:, 1]));  y2i = {v: i for i, v in enumerate(ys)}
        ts = sorted(set(coords[:, 2]));  t2i = {v: i for i, v in enumerate(ts)}
        det_idx = np.array([
            [t2i[coords[i, 2]], y2i[coords[i, 1]], x2i[coords[i, 0]]]
            for i in range(self.num_detectors)
        ], dtype=np.int64)
        self.T, self.H, self.W = len(ts), len(ys), len(xs)
        self.n_grid_channels = 1
        self.n_scalars = 0
        self.grid_dim = self.n_grid_channels * self.T * self.H * self.W

        flat_idx = (det_idx[:, 0] * self.H * self.W
                    + det_idx[:, 1] * self.W
                    + det_idx[:, 2])
        self.register_buffer(
            "_flat_idx", torch.from_numpy(flat_idx).long(), persistent=False
        )

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": self.n_grid_channels,
                "n_scalars": self.n_scalars}

    def cpu_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        syn = batch["syndromes"]
        if not isinstance(syn, torch.Tensor):
            syn = torch.as_tensor(syn)
        return {"ch_syn": syn.float()}

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ch_syn = batch_data["ch_syn"]
        B = ch_syn.size(0)
        device = ch_syn.device

        grid = torch.zeros(B, self.n_grid_channels, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        return grid.reshape(B, -1)


# ─────────────────────────────────────────────────────────────────────────
# Syndrome3DFilteredPreprocessor — syndrome-only input + gap-based filtering
# ─────────────────────────────────────────────────────────────────────────
# Computes MWPM/gap internally (for filtering) but only outputs the syndrome
# volume. Used for fine-tuning the syndrome-only baseline on the routed
# sub-distribution (gap < threshold).
# ─────────────────────────────────────────────────────────────────────────

@register_preprocessor("syndrome_3d_filtered")
class Syndrome3DFilteredPreprocessor(GapRoutedCNNPreprocessor):
    """Syndrome-only preprocessor with gap-based filtering.

    Inherits MWPM setup from GapRoutedCNNPreprocessor (used only for gap
    computation). Output is 1 channel (syndrome) + 0 scalars + `_keep_mask`.
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        kwargs = (config.model.preprocessor or {}).get("kwargs", {})
        return cls(circuit=circuit, **kwargs)

    def __init__(self, circuit, gap_threshold: float = 2.0, **kwargs):
        super().__init__(circuit=circuit, **kwargs)
        self.gap_threshold = float(gap_threshold)
        # Override channel counts — model sees only the syndrome.
        self.n_grid_channels = 1
        self.n_scalars = 0
        self.grid_dim = 1 * self.T * self.H * self.W

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"T": self.T, "H": self.H, "W": self.W,
                "n_grid_channels": 1, "n_scalars": 0}

    def cpu_transform(self, batch):
        full = super().cpu_transform(batch)  # has MWPM byproducts + gap
        gaps = full["scalars"][:, 0]
        return {
            "ch_syn":     full["ch_syn"],
            "_keep_mask": (gaps < self.gap_threshold),
        }

    def gpu_transform(self, batch_data):
        ch_syn = batch_data["ch_syn"]
        B = ch_syn.size(0)
        device = ch_syn.device
        grid = torch.zeros(B, 1, self.T * self.H * self.W,
                           device=device, dtype=torch.float32)
        flat_idx = self._flat_idx.to(device)
        grid[:, 0, flat_idx] = ch_syn.float()
        return grid.reshape(B, -1)
