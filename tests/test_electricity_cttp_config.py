from pathlib import Path

from contsg.config.schema import ExperimentConfig, CTTPModelConfig
from contsg.config.model_validation import validate_model_config


def test_electricity_cttp_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = validate_model_config(ExperimentConfig.from_yaml(
        root / 'configs/cttp/cttp_electricity_v3.yaml')).config
    assert isinstance(config.model, CTTPModelConfig)
    assert config.data.name == 'electricity_15min_semisynth_morph'
    assert config.data.seq_length == 128 and not config.data.normalize
    assert config.model.text_encoding == 'online'
    assert config.model.coemb_dim == 512
    assert config.seed == 42
    assert config.train.epochs == 500
    assert config.train.stages
    saved = tmp_path / 'config.yaml'
    config.to_yaml(saved)
    restored = validate_model_config(ExperimentConfig.from_yaml(saved)).config
    assert restored.model.model_dump() == config.model.model_dump()