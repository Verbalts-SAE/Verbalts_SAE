import json

import numpy as np
import pytest

from scripts.generate_weather_three_stage_dataset import BOUNDS, CHANNELS, inject, make_backgrounds, save_split


def source():
    t = np.arange(3600)
    return np.column_stack((10 + t / 100, 5 + np.sin(t), 1000 + t / 1000)), np.datetime64("2020-01-01") + t * np.timedelta64(10, "m")


def test_single_channel_contiguous_reproducible():
    values, times = source()
    before = values.copy()
    backgrounds, records, _ = make_backgrounds(values, times, (0, 3600), 9, 42)
    np.testing.assert_array_equal(backgrounds, make_backgrounds(values, times, (0, 3600), 9, 42)[0])
    assert backgrounds.shape == (9, 36)
    assert min(np.diff(sorted(r["source_start"] for r in records))) >= 36
    assert [sum(r["channel"] == c for r in records) for c in CHANNELS] == [3, 3, 3]
    for background, r in zip(backgrounds, records):
        raw = values[r["source_start"]:r["source_stop"], CHANNELS.index(r["channel"])]
        np.testing.assert_allclose(background, (raw - raw[0]) * r["scale"])
        for boundary in (12, 24):
            assert background[boundary] - background[boundary - 1] == pytest.approx((raw[boundary] - raw[boundary - 1]) * r["scale"])
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("stage", range(3))
@pytest.mark.parametrize("name", ["single_peak", "double_peaks", "sag"])
def test_stage_locality(stage, name):
    labels = ["nothing"] * 3
    labels[stage] = name
    background = np.arange(36, dtype=float)
    curve, residual = inject(background, labels)
    start, stop = BOUNDS[stage]
    assert np.count_nonzero(residual[:start]) == np.count_nonzero(residual[stop:]) == 0
    assert residual[start] == residual[stop - 1] == 0
    np.testing.assert_allclose(curve - residual, background)
    local = residual[start:stop]
    if name == "double_peaks":
        assert np.sum((local[1:-1] > local[:-2]) & (local[1:-1] > local[2:])) == 2
    elif name == "sag":
        assert local.min() < 0 and local.max() == 0
    else:
        assert local.max() > 0 and local.min() == 0


def test_saved_decomposition_balance_and_reproducibility(tmp_path):
    values, times = source()
    backgrounds, records, _ = make_backgrounds(values, times, (0, 3600), 3, 2)
    for directory in (tmp_path / "a", tmp_path / "b"):
        shapes = save_split(directory, backgrounds, records, 1.2, 0.08, 99)
        assert shapes == {"combined": [81, 36, 1], "isolated": [27, 36, 1]}
        for prefix, repeats in (("combined", 27), ("isolated", 9)):
            arrays = {s: np.load(directory / f"{prefix}_{s}.npy") for s in ("samples", "residuals", "morphology_residuals", "noise")}
            np.testing.assert_allclose(arrays["samples"], np.repeat(backgrounds, repeats, axis=0)[..., None] + arrays["morphology_residuals"] + arrays["noise"], atol=1e-12)
            np.testing.assert_allclose(arrays["residuals"], arrays["morphology_residuals"] + arrays["noise"])
            labels = np.load(directory / f"{prefix}_labels.npy")
            assert labels.shape == (3 * repeats, 3)
            assert np.all(np.unique(labels, axis=0, return_counts=True)[1] == 3)
            assert len(json.loads((directory / f"{prefix}_records.json").read_text())) == 3 * repeats
    np.testing.assert_array_equal(np.load(tmp_path / "a/combined_samples.npy"), np.load(tmp_path / "b/combined_samples.npy"))


def test_invalid_input_and_missing_time():
    values, times = source()
    times[1] = times[0]
    with pytest.raises(ValueError, match="Need"):
        make_backgrounds(values, times, (0, 36), 1, 2)
    with pytest.raises(ValueError):
        inject(np.zeros(128), ["sag"] * 3)
    with pytest.raises(ValueError):
        inject(np.zeros(36), ["sag"] * 4)