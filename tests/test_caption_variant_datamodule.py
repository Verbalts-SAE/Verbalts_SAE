"""Tests for selecting sparse/full caption files."""

import json

import numpy as np

from contsg.data.datamodule import TimeSeriesDataset


def test_time_series_dataset_selects_caption_variant(tmp_path) -> None:
    np.save(tmp_path / "train_ts.npy", np.zeros((2, 128, 1), dtype=np.float32))
    np.save(tmp_path / "train_text_caps_sparse.npy", np.asarray(["s0", "s1"]))
    np.save(tmp_path / "train_text_caps_full.npy", np.asarray(["f0", "f1"]))
    (tmp_path / "meta.json").write_text(json.dumps({"n_var": 1, "seq_length": 128}))
    sparse = TimeSeriesDataset(tmp_path, normalize=False, caption_variant="sparse")
    full = TimeSeriesDataset(tmp_path, normalize=False, caption_variant="full")
    assert sparse[0]["cap"] == "s0"
    assert full[0]["cap"] == "f0"


def test_time_series_dataset_selects_precomputed_variant_embedding(tmp_path) -> None:
    np.save(tmp_path / "train_ts.npy", np.zeros((2, 128, 1), dtype=np.float32))
    np.save(tmp_path / "train_text_caps_sparse.npy", np.asarray(["s0", "s1"]))
    expected = np.arange(16, dtype=np.float32).reshape(2, 8)
    np.save(tmp_path / "train_cap_emb_sparse_qwen3-embedding-0.6b_8.npy", expected)
    (tmp_path / "meta.json").write_text(json.dumps({"n_var": 1, "seq_length": 128}))

    dataset = TimeSeriesDataset(tmp_path, normalize=False, caption_variant="sparse")

    np.testing.assert_array_equal(dataset[1]["cap_emb"].numpy(), expected[1])