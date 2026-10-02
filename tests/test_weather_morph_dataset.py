import importlib.util

import numpy as np
import pandas as pd
import pytest

from scripts.generate_weather_morph_dataset import (
    CHANNELS, ROOT, make_backgrounds, prepare_source, save_split, select_scale, source_intervals,
)


def source(size=3600):
    rng = np.random.default_rng(12)
    values = rng.normal(size=(size, 3)) + [10, 5, 1000]
    times = np.datetime64("2020-01-01") + np.arange(size) * np.timedelta64(10, "m")
    return values, times


def test_backgrounds_reproducible_disjoint_and_aligned():
    values, times = source()
    backgrounds, records, _ = make_backgrounds(values, times, (0, len(values)), 5, 42)
    np.testing.assert_array_equal(backgrounds, make_backgrounds(values, times, (0, len(values)), 5, 42)[0])
    assert backgrounds.shape == (5, 128)
    starts = []
    for background, record in zip(backgrounds, records):
        assert set(s["channel"] for s in record["segments"]) == set(CHANNELS)
        for s in record["segments"]:
            starts.append(s["source_start"])
            raw = values[s["source_start"]:s["source_stop"], CHANNELS.index(s["channel"])]
            np.testing.assert_allclose(background[s["output_start"]:s["output_stop"]],
                                       (raw - raw[0]) * s["scale"] + s["anchor"])
    assert np.min(np.diff(sorted(starts))) >= 36
    for b in (36, 72, 108):
        np.testing.assert_array_equal(backgrounds[:, b], backgrounds[:, b - 1])


def test_time_partitions_do_not_overlap():
    intervals = list(source_intervals(3600).values())
    assert all(right[0] - left[1] == 36 for left, right in zip(intervals, intervals[1:]))


def test_prepare_source_excludes_duplicate_times_and_preserves_row_ids():
    frame = pd.DataFrame({"Date Time": ["01.01.2020 00:30:00", "01.01.2020 00:10:00",
                                       "01.01.2020 00:20:00", "01.01.2020 00:20:00"]})
    cleaned = prepare_source(frame)
    assert cleaned["original_row"].tolist() == [1, 0]
    assert cleaned["timestamp"].is_monotonic_increasing


def test_insufficient_and_invalid_sources():
    values, times = source(144)
    with pytest.raises(ValueError, match="Need"):
        make_backgrounds(values, times, (0, 144), 2, 0)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="Need"):
        make_backgrounds(values, times, (0, 144), 1, 0)
    values, times = source(144)
    times[1] = times[0]
    with pytest.raises(ValueError, match="Need"):
        make_backgrounds(values, times, (0, 144), 1, 0)


def test_scale_matches_approved_preview():
    path = ROOT / "outputs/weather_univariate_preview/normalized_preview.py"
    spec = importlib.util.spec_from_file_location("approved_weather", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(7)
    for raw in (np.ones(36), np.arange(36), rng.normal(size=36)):
        for channel in CHANNELS:
            assert select_scale(raw, channel)[0] == module.select_scale(raw, channel)[0]


def test_saved_split(tmp_path):
    values, times = source()
    backgrounds, records, _ = make_backgrounds(values, times, (0, len(values)), 2, 0)
    shapes = save_split(tmp_path, "train", backgrounds, records, 1.75)
    assert shapes == {"combined": [162, 128, 1], "isolated": [24, 128, 1]}
    curves = np.load(tmp_path / "combined_samples.npy")
    residuals = np.load(tmp_path / "combined_residuals.npy")
    np.testing.assert_allclose(curves - residuals, np.repeat(backgrounds, 81, axis=0)[..., None], atol=1e-12)
    labels = np.load(tmp_path / "combined_labels.npy")
    assert labels.shape == (162, 4)
    assert np.all(np.unique(labels, axis=0, return_counts=True)[1] == 2)