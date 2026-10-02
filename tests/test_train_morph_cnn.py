import numpy as np
import pytest

from sae.train_morph_cnn import load_split


def test_split_preserves_boundaries_and_labels(tmp_path):
    curves = np.arange(256, dtype=np.float32).reshape(2, 128, 1)
    np.save(tmp_path / "train_ts.npy", curves)
    np.save(tmp_path / "train_attrs_idx.npy", np.array([[0, 1, 2], [3, 2, 1]]))
    x, p, v = load_split(tmp_path, "train").tensors
    assert x.shape == (6, 1, 43)
    assert x[2, 0, 0] == 85
    assert x[3, 0, 0] == 128
    assert p.tolist() == [0, 1, 2, 0, 2, 1]
    assert v.tolist() == [0, 0, 0, 1, 0, 0]
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "valid")


def test_rejects_invalid_labels(tmp_path):
    np.save(tmp_path / "train_ts.npy", np.zeros((1, 128, 1)))
    np.save(tmp_path / "train_attrs_idx.npy", np.array([[0, 1, 4]]))
    with pytest.raises(ValueError, match="integers"):
        load_split(tmp_path, "train")