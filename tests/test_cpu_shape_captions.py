import hashlib
import json

import numpy as np
import pytest

from scripts.preview_cpu_background_denoising import denoise_background
from scripts.preview_cpu_shape_captions import local_caption, prepare, trajectory, validate_caption


@pytest.mark.parametrize("levels, expected", [
    ([0, 0, 0], "flat"), ([0, 1, 2], "rise"), ([2, 1, 0], "fall"),
    ([0, 2, 0], "rise_fall"), ([2, 0, 2], "fall_rise"),
    ([0, 2, 2], "rise_flat"), ([2, 0, 0], "fall_flat"),
    ([0, 0, 2], "flat_rise"), ([2, 2, 0], "flat_fall"),
])
def test_trajectory_shapes(levels, expected):
    x = np.interp(np.linspace(0, 1, 128), [0, .5, 1], levels)
    original = x.copy()
    result = trajectory(x)
    assert result["shape"] == expected
    assert trajectory(x + 10)["shape"] == expected
    validate_caption(result["caption"])
    np.testing.assert_array_equal(x, original)


def test_noise_narrow_spike_and_multiple_turns():
    t = np.linspace(0, 1, 128)
    assert trajectory(2 * t + np.random.default_rng(4).normal(0, .03, len(t)))["shape"] == "rise"
    spike = np.zeros(128)
    spike[60] = 2
    assert trajectory(spike)["shape"] == "flat"
    result = trajectory(np.sin(8 * np.pi * t))
    assert result["shape"] == "irregular" and result["review_required"]


def test_local_fixed_and_numeric_free():
    for start, stage in ((12, "beginning"), (51, "middle"), (91, "end")):
        text = local_caption(dict(injected_kind="double_peaks", start=start, stop_exclusive=start + 25), 128)
        assert text == f"The {stage} part has double peaks."
        validate_caption(text)
    assert local_caption(dict(injected_kind="nothing"), 128) == ""
    for bad in ("A peak at 15.", "detail 50%", "high complexity"):
        with pytest.raises(ValueError):
            validate_caption(bad)


@pytest.mark.parametrize("x", [[1, 2], [[1] * 128], [np.nan] * 128])
def test_invalid_input(x):
    with pytest.raises(ValueError):
        trajectory(x)


def test_export_and_input_integrity(tmp_path):
    source, previous, output = (tmp_path / name for name in ("source", "previous", "output"))
    source.mkdir()
    previous.mkdir()
    bg = np.stack([np.linspace(0, 2, 128), np.zeros(128)])
    light = np.array([denoise_background(x, .5) for x in bg])
    residuals = np.zeros_like(bg)
    residuals[0, 23:26] = [.6, 1.2, .6]
    cases = [dict(source_row=i, injected_kind=kind, start=12, stop_exclusive=37)
             for i, kind in enumerate(("single_peak", "nothing"))]
    (source / "manifest.json").write_text(json.dumps(dict(config=dict(amplitude=1.2), cases=cases)))
    (previous / "captions.json").write_text(json.dumps([dict(source_row=i, caption="Old caption.") for i in range(2)]))
    np.savez(source / "preview_arrays.npz", source_rows=np.arange(2), adaptive_backgrounds=bg,
             light_backgrounds=light, residuals=residuals, light_samples=light + residuals)
    path = source / "preview_arrays.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = prepare(source, previous, output)
    assert report["source_unchanged"] and report["shape_counts"] == {"rise": 1, "flat": 1}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    caps = np.load(output / "text_caps.npy")
    assert caps.shape == (2, 1)
    assert caps[0, 0] == "The beginning part has a single peak. The background generally rises across the sequence."
    assert caps[1, 0] == "The background stays roughly level."
    assert (output / "comparison_01_02.png").exists()
    with pytest.raises(FileExistsError):
        prepare(source, previous, output)