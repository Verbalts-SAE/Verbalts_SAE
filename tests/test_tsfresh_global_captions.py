"""Tests for tsfresh global features and combined morphology captions."""

import json

import numpy as np

from contsg.data.datasets.tsfresh_global import (
    CAPTION_FEATURE_NAMES,
    FEATURE_NAMES,
    combine_global_and_local_caption,
    extract_global_features,
    fit_caption_thresholds,
    make_global_description,
)
from contsg.data.datasets.semisynth_morph import make_caption
from sae.shapes import parse_segment_shapes
from scripts.add_tsfresh_global_captions import SPLITS, augment


def test_extract_global_features_has_stable_order_and_finite_values():
    x = np.linspace(0, 1, 32)
    curves = np.stack((x, x[::-1], np.sin(2 * np.pi * x)))[..., None]
    features = extract_global_features(curves, n_jobs=1)
    assert features.shape == (3, len(FEATURE_NAMES))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert features[0, 0] > 0
    assert features[1, 0] < 0


def test_global_description_uses_train_tertiles():
    train_features = np.asarray(
        [[-2, 1, 1, 1, 1], [0, 2, 2, 2, 2], [2, 3, 3, 3, 3]], dtype=np.float32
    )
    thresholds = fit_caption_thresholds(train_features)
    low = make_global_description(train_features[0], thresholds)
    high = make_global_description(train_features[-1], thresholds)
    assert "a downward trend" in low
    assert "low variability" in low
    assert "low structural complexity" in low
    assert "weak short-term persistence" in low
    assert "an upward trend" in high
    assert "high variability" in high
    assert "high structural complexity" in high
    assert "strong short-term persistence" in high
    assert CAPTION_FEATURE_NAMES == FEATURE_NAMES
    assert not any(char.isdigit() for char in low + high)


def test_combined_caption_preserves_fine_grained_parser_labels():
    local = make_caption(("single_peak", "double_peaks", "sag"))
    global_description = "The time series has an upward overall trend."
    combined = combine_global_and_local_caption(global_description, local)
    assert combined.startswith("The time series")
    assert "Fine-grained morphology:" not in combined
    assert combined == f"{global_description} {local.strip().replace('.The ', '. The ')}"
    assert ".The " not in combined
    assert parse_segment_shapes(combined) == ("single peak", "double peaks", "sag")


def test_extract_global_features_rejects_non_finite_curves():
    curves = np.zeros((2, 32), dtype=np.float32)
    curves[0, 0] = np.nan
    try:
        extract_global_features(curves, n_jobs=1)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("expected non-finite curves to be rejected")


def test_augment_writes_only_canonical_captions_idempotently(tmp_path):
    x = np.linspace(0, 1, 32, dtype=np.float32)
    curves = np.stack((x, x[::-1], np.sin(2 * np.pi * x)))[..., None]
    local = make_caption(("single_peak", "nothing", "sag"))
    for split in SPLITS:
        np.save(tmp_path / f"{split}_ts.npy", curves)
        np.save(tmp_path / f"{split}_text_caps.npy", np.asarray([[local]] * len(curves)))
        np.save(tmp_path / f"{split}_caps.npy", np.asarray([[local]] * len(curves)))
        np.save(tmp_path / f"{split}_text_caps_tsfresh.npy", np.asarray(["obsolete"] * len(curves)))
    (tmp_path / "meta.json").write_text(json.dumps({"version": 3}), encoding="utf-8")
    (tmp_path / "validation.json").write_text(
        json.dumps({"morphology_condition_file": "obsolete"}), encoding="utf-8"
    )
    (tmp_path / "DATASET_CARD.md").write_text("# test dataset\n", encoding="utf-8")

    summary = augment(tmp_path, n_jobs=1)
    first = np.load(tmp_path / "train_text_caps.npy")
    augment(tmp_path, n_jobs=1)

    assert summary["caption_variant"] == "base"
    assert summary["caption_file"] == "{split}_text_caps.npy"
    assert "global_only" not in summary
    validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert validation["caption_contains_morphology"] is True
    assert "morphology_condition_file" not in validation
    for split in SPLITS:
        assert not (tmp_path / f"{split}_caps.npy").exists()
        assert not (tmp_path / f"{split}_text_caps_tsfresh.npy").exists()
    assert np.load(tmp_path / "train_tsfresh_global.npy").shape == (3, len(FEATURE_NAMES))
    combined = np.load(tmp_path / "train_text_caps.npy").reshape(-1)
    np.testing.assert_array_equal(first, combined)
    assert combined[0].startswith("The time series")
    assert all("Fine-grained morphology:" not in caption for caption in combined)
    assert parse_segment_shapes(combined[0]) == ("single peak", "nothing", "sag")
    card = (tmp_path / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert card.count("## tsfresh global + fine-grained captions") == 1