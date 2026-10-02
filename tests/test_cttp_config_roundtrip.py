"""Regression tests for model-specific CTTP experiment provenance."""

from pathlib import Path

from contsg.config.model_validation import validate_model_config
from contsg.config.schema import CTTPModelConfig, ExperimentConfig


def test_cttp_specific_fields_survive_yaml_roundtrip(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/cttp/cttp_synth-u.yaml"
    config = validate_model_config(ExperimentConfig.from_yaml(source)).config
    assert isinstance(config.model, CTTPModelConfig)
    assert config.model.coemb_dim == 512
    assert config.model.patch_len == 4

    saved = tmp_path / "config.yaml"
    config.to_yaml(saved)
    reloaded = validate_model_config(ExperimentConfig.from_yaml(saved)).config
    assert isinstance(reloaded.model, CTTPModelConfig)
    assert reloaded.model.model_dump() == config.model.model_dump()


def test_cttp_specific_fields_appear_in_human_readable_yaml() -> None:
    source = Path(__file__).resolve().parents[1] / "configs/cttp/cttp_synth-u.yaml"
    config = validate_model_config(ExperimentConfig.from_yaml(source)).config
    rendered = config.model_dump_yaml()
    assert "coemb_dim: 512" in rendered
    assert "patch_len: 4" in rendered
    assert "pretrain_model_path: ./checkpoints/longclip" in rendered