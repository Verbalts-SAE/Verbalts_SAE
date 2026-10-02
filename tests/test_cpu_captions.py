import json

import numpy as np
import pytest

from scripts.preview_cpu_captions import (
    features, fit_preview_thresholds, global_caption, local_caption, prepare, validate_caption,
)
from scripts.preview_cpu_background_denoising import denoise_background


def test_local_positions_and_nothing():
    for index, (start, stage) in enumerate(((12, "beginning"), (51, "middle"), (91, "end"))):
        case = dict(injected_kind="double_peaks", start=start, stop_exclusive=start + 25)
        assert f"The {stage} part has double peaks." == local_caption(case, 128, -index)
    assert local_caption(dict(injected_kind="nothing"), 128) == ""
    with pytest.raises(ValueError):
        local_caption(dict(injected_kind="unknown"), 128)


def test_direction_and_constant_background():
    matrix = features(np.stack([np.arange(128), -np.arange(128), np.ones(128)]))
    thresholds = fit_preview_thresholds(matrix)
    assert "rises" in global_caption(matrix[0], thresholds)
    assert "falls" in global_caption(matrix[1], thresholds)
    text = global_caption(matrix[2], thresholds)
    assert "mostly flat" in text
    assert "persistence" not in text
    assert "nan" not in text.lower()


def test_export_and_reject_wrong_retention(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bg = np.random.default_rng(5).normal(size=(2, 128))
    light = np.array([denoise_background(x, .5) for x in bg])
    residuals = np.zeros_like(bg)
    residuals[0, 20] = 1.2
    cases = [dict(source_row=i, injected_kind=kind, start=12, stop_exclusive=37)
             for i, kind in enumerate(("single_peak", "nothing"))]
    (source / "manifest.json").write_text(json.dumps(dict(config=dict(amplitude=1.2), cases=cases)))
    arrays = dict(source_rows=np.arange(2), adaptive_backgrounds=bg,
                  light_backgrounds=light, light_samples=light + residuals, residuals=residuals)
    np.savez(source / "preview_arrays.npz", **arrays)
    output = tmp_path / "output"
    report = prepare(source, output)
    assert report["threshold_scope"] == "preview_only_not_for_training"
    caps = np.load(output / "text_caps.npy")
    assert caps.shape == (2, 1)
    assert caps[0, 0].startswith("The beginning part has a single peak")
    assert caps[1, 0].startswith("The series")
    for caption in caps[:, 0]:
        validate_caption(caption)
        assert "Its global pattern has" in caption
    np.testing.assert_allclose(np.load(output / "background_features.npy"), features(light))
    with pytest.raises(FileExistsError):
        prepare(source, output)
    arrays["light_backgrounds"] = bg
    np.savez(source / "preview_arrays.npz", **arrays)
    with pytest.raises(AssertionError):
        prepare(source, tmp_path / "bad_output")


@pytest.mark.parametrize("text", ["The slope is 0.3.", "Detail is 50%.",
                                       "The amplitude is high.", "tsfresh features"])
def test_reject_numeric_or_processing_caption(text):
    with pytest.raises(ValueError):
        validate_caption(text)