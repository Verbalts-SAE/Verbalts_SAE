import numpy as np
import pytest

from scripts.prepare_three_model_tuning import MODELS, LEARNING_RATES, candidate_config
from scripts.train_three_model_candidate import exemplar_index, install_train_only_exemplars


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("lr", LEARNING_RATES)
def test_candidate_schema(model, lr):
    from contsg.config.schema import ExperimentConfig
    cfg = candidate_config(model, lr)
    parsed = ExperimentConfig(**cfg)
    assert parsed.model.name == model
    assert parsed.data.normalize is False
    assert cfg["seed"] == 42
    assert all(s["lr"] == lr for s in cfg["train"]["stages"])
    assert cfg["train"]["batch_size"] * cfg["train"]["accumulate_grad_batches"] == 256
    assert candidate_config(model, lr, True)["train"]["epochs"] == len(cfg["train"]["stages"])


def test_train_only_exemplar(tmp_path, monkeypatch):
    from contsg.data.datamodule import TimeSeriesDataset
    original = TimeSeriesDataset.__getitem__
    monkeypatch.setattr(TimeSeriesDataset, "__getitem__", original)
    train = np.arange(7 * 128, dtype=np.float32).reshape(7, 128, 1)
    np.save(tmp_path / "train_ts.npy", train)
    np.save(tmp_path / "valid_ts.npy", np.full((3, 128, 1), -999, dtype=np.float32))
    install_train_only_exemplars()
    dataset = TimeSeriesDataset(tmp_path, split="valid", normalize=False, provide_bridge_example=True)
    for i in range(3):
        item = dataset[i]
        np.testing.assert_array_equal(item["bridge_example_ts"].numpy(), train[exemplar_index(i, 7)])
        assert (item["ts"] == -999).all()
    with pytest.raises(ValueError):
        exemplar_index(0, 0)