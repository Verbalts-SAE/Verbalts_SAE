import numpy as np
import pytest

from scripts.prepare_weather36_captions import features, fit_thresholds, background_caption, local_caption
from scripts.prepare_weather36_tuning import candidate, MODELS


def test_features_and_direction():
    x = np.arange(36, dtype=float)
    rows = features(np.stack([x, -x, x * 0]))
    np.testing.assert_allclose(rows[:, 0], [1, -1, 0])
    thresholds = fit_thresholds(rows)
    assert "rises" in background_caption(rows[0], thresholds)
    assert "falls" in background_caption(rows[1], thresholds)
    assert "mostly flat" in background_caption(rows[2], thresholds)
    assert np.isnan(rows[2, 4])


def test_morphology_first():
    assert local_caption(["single_peak", "double_peaks", "sag"]) == (
        "The beginning part has a single peak. The middle part has double peaks. The end part has a sag.")
    assert local_caption(["nothing", "sag", "nothing"]) == "The middle part has a sag."


def test_dataset_registration():
    from contsg.registry import Registry
    from contsg.data.datasets.weather36 import Weather36DataModule
    assert Registry.get_dataset("weather_three_stage_36_morph") is Weather36DataModule


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("lr", [1e-4, 3e-4, 1e-3])
def test_config(model, lr):
    from contsg.config.schema import ExperimentConfig
    cfg = ExperimentConfig(**candidate(model, lr))
    assert cfg.data.seq_length == 36 and cfg.data.normalize is False
    assert cfg.train.batch_size * cfg.train.accumulate_grad_batches == 256
    assert all(stage.lr == lr for stage in cfg.train.stages)


@pytest.mark.parametrize("length, expected", [(36, 40), (128, 128)])
def test_diffusets_encoder_length(length, expected):
    import torch
    from contsg.models.diffusets import DiffuSETS
    class EncoderProbe:
        def encoder(self, x):
            return x
    x = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
    result = DiffuSETS.encode(EncoderProbe(), x)
    assert result.shape == (1, expected, 1)
    torch.testing.assert_close(result[:, :length], x)
    assert result[0, -1, 0] == length - 1