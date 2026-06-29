"""ExperimentConfig.from_yaml의 SimulationConfig 검증 동작.

commit 4e5c413에서 simulation: Dict → SimulationConfig 타입화하면서
yaml 오타 catch 동작 추가됨. 회귀 방지.
"""
import textwrap
import pytest

from qec_sim.config.schema import ExperimentConfig


# 최소 정상 yaml — 다른 테스트의 베이스로 사용.
_BASE_YAML = """
code: {name: "surface_code", distance: 3, rounds: 3}
noise: {p_gate: 0.001, p_meas: 0.005, p_corr: 0.0}
training:
  data_mode: "online"
  epochs: 1
  batch_size: 256
  train_steps: 100
  val_steps: 20
  output_dir: "./tmp"
  optimizer: {name: "Adam", kwargs: {lr: 0.001}}
  criterion: {name: "bce_with_logits", kwargs: {}}
  early_stopping: {patience: 10}
model: {name: "jung_cnn_low_soft", kwargs: {}}
decoder: {name: "neural_decoder", weight_path: "", model_kwargs: {}}
simulation:
  backend: "pauli_plus"
  pauli_plus: {p: 0.001}
"""


def _write(tmp_path, yaml_text):
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    return p


def test_loads_valid_config(tmp_path):
    """정상 yaml은 깨끗하게 로드되고 simulation 필드가 typed."""
    cfg = ExperimentConfig.from_yaml(str(_write(tmp_path, _BASE_YAML)))
    assert cfg.simulation.backend == "pauli_plus"
    assert cfg.simulation.pauli_plus == {"p": 0.001}
    assert cfg.simulation.shots is None  # optional


def test_missing_backend_raises(tmp_path):
    """simulation.backend 누락 시 KeyError."""
    yaml_no_backend = _BASE_YAML.replace('backend: "pauli_plus"', '').replace(
        'pauli_plus: {p: 0.001}', 'pauli_plus: {p: 0.001}'
    )
    # backend 라인 제거된 yaml
    yaml_no_backend = textwrap.dedent("""\
        code: {name: "surface_code", distance: 3, rounds: 3}
        noise: {p_gate: 0.001, p_meas: 0.005, p_corr: 0.0}
        training:
          data_mode: "online"
          epochs: 1
          batch_size: 256
          output_dir: "./tmp"
          optimizer: {name: "Adam", kwargs: {lr: 0.001}}
          criterion: {name: "bce_with_logits", kwargs: {}}
          early_stopping: {patience: 10}
        model: {name: "jung_cnn_low_soft", kwargs: {}}
        decoder: {name: "neural_decoder", weight_path: "", model_kwargs: {}}
        simulation:
          shots: 1000
    """)
    with pytest.raises(KeyError):
        ExperimentConfig.from_yaml(str(_write(tmp_path, yaml_no_backend)))


def test_unknown_pauli_plus_key_raises(tmp_path):
    """pauli_plus에 PauliPlusNoiseParams 필드명 아닌 키가 있으면 KeyError."""
    yaml_typo = _BASE_YAML.replace(
        'pauli_plus: {p: 0.001}',
        'pauli_plus: {p: 0.001, p_corsstalk: 1.0e-3}'  # 의도적 오타
    )
    with pytest.raises(KeyError, match="알 수 없는"):
        ExperimentConfig.from_yaml(str(_write(tmp_path, yaml_typo)))


def test_stim_backend_no_pauli_plus_required(tmp_path):
    """backend=stim이면 pauli_plus 블록 없어도 정상 로드."""
    yaml_stim = textwrap.dedent("""\
        code: {name: "surface_code", distance: 3, rounds: 3}
        noise: {p_gate: 0.001, p_meas: 0.005, p_corr: 0.0}
        training:
          data_mode: "online"
          epochs: 1
          batch_size: 256
          train_steps: 100
          val_steps: 20
          output_dir: "./tmp"
          optimizer: {name: "Adam", kwargs: {lr: 0.001}}
          criterion: {name: "bce_with_logits", kwargs: {}}
          early_stopping: {patience: 10}
        model: {name: "jung_cnn_low_soft", kwargs: {}}
        decoder: {name: "neural_decoder", weight_path: "", model_kwargs: {}}
        simulation:
          backend: "stim"
    """)
    cfg = ExperimentConfig.from_yaml(str(_write(tmp_path, yaml_stim)))
    assert cfg.simulation.backend == "stim"
    assert cfg.simulation.pauli_plus is None
