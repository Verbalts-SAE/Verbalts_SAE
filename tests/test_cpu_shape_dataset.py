import json

import numpy as np
import pytest

from scripts.build_cpu_shape_dataset import build


def test_build_alignment_and_isolation(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.random.default_rng(9).normal(size=(120, 128)))
    output = tmp_path / "dataset"
    report = build(source, output)
    assert report["status"] == "PASS"
    all_ids = []
    for split, count in (("train", 96), ("valid", 12), ("test", 12)):
        ts = np.load(output / f"{split}_ts.npy")
        bg = np.load(output / f"{split}_backgrounds.npy")
        residual = np.load(output / f"{split}_residuals.npy")
        np.testing.assert_array_equal(ts[..., 0], (bg + residual).astype(np.float32))
        assert ts.shape == (count, 128, 1)
        records = json.loads((output / f"{split}_records.json").read_text())
        caps = np.load(output / f"{split}_text_caps.npy")
        assert caps[:, 0].tolist() == [r["caption"] for r in records]
        all_ids.extend(np.load(output / f"{split}_source_indices.npy").tolist())
        assert len(set(report["splits"][split]["kinds"].values())) == 1
    assert sorted(all_ids) == list(range(120))
    with pytest.raises(FileExistsError):
        build(source, output)


def test_duplicate_background_rejected(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((20, 128)))
    with pytest.raises(ValueError, match="Duplicate"):
        build(source, tmp_path / "dataset")