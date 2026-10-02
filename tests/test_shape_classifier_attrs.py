"""Regression tests for authoritative morphology-attribute labels."""

import numpy as np
import torch

from sae.shape_classifier import load_activation_cache


def test_load_activation_cache_accepts_three_column_attrs(tmp_path) -> None:
    cache = tmp_path / "cache.pt"
    attrs = tmp_path / "attrs.npy"
    torch.save(
        {
            "activations": torch.randn(6, 4),
            "t_values": torch.tensor([5, 5, 5, 9, 9, 9]),
            "audit": {"tokens_per_sample": 3},
            "t_range": [5, 45],
        },
        cache,
    )
    expected = np.asarray([[0, 1, 2], [3, 2, 1]], dtype=np.int64)
    np.save(attrs, expected)

    samples, timesteps, labels, _ = load_activation_cache(
        cache, attrs, labels_from_attrs=True
    )

    assert samples.shape == (2, 3, 4)
    torch.testing.assert_close(timesteps, torch.tensor([5, 9]))
    torch.testing.assert_close(labels, torch.from_numpy(expected))