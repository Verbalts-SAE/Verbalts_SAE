import numpy as np
import pytest

from sae.eval_morph_steering import configurations, global_levels, evaluation_indices
from contsg.data.datasets.tsfresh_global import FEATURE_NAMES


def test_global_levels_boundaries():
    thresholds = {name: (1., 2.) for name in FEATURE_NAMES}
    features = np.repeat(np.array([0., 1., 1.5, 2., 3.])[:, None], 5, axis=1)
    expected = np.repeat(np.array([0, 1, 1, 1, 2])[:, None], 5, axis=1)
    np.testing.assert_array_equal(global_levels(features, thresholds), expected)


def test_small_grid_has_baselines_and_reference():
    variants = configurations()
    assert len(variants) == 14
    assert len({v["name"] for v in variants}) == len(variants)
    assert variants[:2] == [{"name": "pure"}, {"name": "sae"}]
    assert any(v.get("topk") == 32 and v.get("eta") == 5120 and
               v.get("gamma") == 6 for v in variants)
    assert any(v.get("topk") == 8 for v in variants)
    assert any(v.get("topk") == 128 for v in variants)
    assert any(v.get("eta") == 8192 for v in variants)
    assert any(v.get("gamma") == 9 for v in variants)


def test_full_split_and_unchanged_validation_sampling():
    expected = np.random.default_rng(2026).permutation(2000)
    np.testing.assert_array_equal(evaluation_indices(2000, 0), expected)
    np.testing.assert_array_equal(evaluation_indices(2000, 128, 768), expected[768:896])
    assert len(set(evaluation_indices(2000, 0))) == 2000
    for count, offset in [(0, 1), (-1, 0), (2001, 0), (128, -1)]:
        with pytest.raises(ValueError):
            evaluation_indices(2000, count, offset)