import json

import numpy as np

from contsg.data.datasets.semisynth_morph import MORPHOLOGY_NAMES, inject
from sae.shapes import parse_segment_shapes
from scripts.rewrite_electricity_v3_global_captions import (
    SPLITS,
    extract_background,
    make_caption,
    make_pattern_sentence,
    rebuild_residual,
    rewrite,
    trend_level,
)


def _pattern_thresholds():
    return {
        "linear_trend_slope": (-0.5, 0.5),
        "standard_deviation": (0.7, 0.9),
        "mean_abs_change": (0.13, 0.15),
        "cid_ce_normalized": (2.3, 2.7),
        "autocorrelation_lag_1": (0.96, 0.98),
    }


def _make_curve(attrs, rng, background=None):
    """Build one ts via inject so tests exercise the real composition path."""
    if background is None:
        background = rng.normal(0.0, 1.0, 128)
    ts, residual = inject(background, tuple(MORPHOLOGY_NAMES[a] for a in attrs))
    return ts, residual


def test_rebuild_residual_matches_inject():
    rng = np.random.default_rng(0)
    for attrs in ([1, 2, 3], [0, 0, 0], [3, 1, 2]):
        _, residual = _make_curve(attrs, rng)
        rebuilt = rebuild_residual(np.asarray(attrs, dtype=np.int64))
        np.testing.assert_allclose(rebuilt, residual, atol=1e-6)


def test_extract_background_recovers_pure_base():
    rng = np.random.default_rng(1)
    background = rng.normal(0.0, 1.0, 128)
    attrs = np.asarray([1, 2, 3], dtype=np.int64)
    ts, _ = _make_curve(attrs, rng, background=background)
    recovered = extract_background(ts, attrs)
    # base = robust_normalize(background) + residual; ts - residual must equal it.
    from contsg.data.datasets.semisynth_morph import robust_normalize

    base, _, _ = robust_normalize(background)
    np.testing.assert_allclose(recovered, base, atol=1e-6)


def test_make_caption_is_morphology_first_then_global():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    thresholds = _pattern_thresholds()
    feats = np.asarray([0.0, 0.95, 0.16, 2.8, 0.99], dtype=np.float64)
    caption = make_caption(
        np.asarray([3, 0, 1]), slope=0.9, trend_bounds=bounds,
        pattern_features=feats, pattern_thresholds=thresholds, sample_index=0,
    )
    assert caption == (
        "The beginning part has a sag. A single peak at the end area. "
        "The series rises gradually overall. Its global pattern has fairly high "
        "variability, fairly brisk point-to-point changes, fairly pronounced structural complexity, "
        "and strong short-term persistence."
    )
    assert "nothing" not in caption.lower()
    assert caption.index("sag") < caption.index("overall") < caption.index("global pattern")


def test_make_caption_all_nothing_keeps_only_global_sentences():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    thresholds = _pattern_thresholds()
    feats = np.asarray([-3.0, 0.5, 0.1, 2.0, 0.5], dtype=np.float64)
    caption = make_caption(
        np.asarray([0, 0, 0]), slope=-3.0, trend_bounds=bounds,
        pattern_features=feats, pattern_thresholds=thresholds, sample_index=0,
    )
    assert caption == (
        "The series falls steeply overall. Its global pattern has low "
        "variability, smooth point-to-point changes, low structural complexity, "
        "and weak short-term persistence."
    )
    assert parse_segment_shapes(caption) == ("nothing", "nothing", "nothing")


def test_make_caption_round_trips_through_parser():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    thresholds = _pattern_thresholds()
    feats = np.asarray([0.0, 0.8, 0.14, 2.5, 0.97], dtype=np.float64)
    attrs = np.asarray([1, 2, 3])
    caption = make_caption(
        attrs, slope=0.0, trend_bounds=bounds, pattern_features=feats,
        pattern_thresholds=thresholds, sample_index=4,
    )
    parsed = parse_segment_shapes(caption)
    expected = tuple(MORPHOLOGY_NAMES[int(v)].replace("_", " ") for v in attrs)
    assert parsed == expected


def test_make_pattern_sentence_buckets():
    thresholds = _pattern_thresholds()
    low = make_pattern_sentence(
        np.asarray([0.0, 0.5, 0.1, 2.0, 0.5]), thresholds
    )
    mid = make_pattern_sentence(
        np.asarray([0.0, 0.8, 0.14, 2.5, 0.97]), thresholds
    )
    high = make_pattern_sentence(
        np.asarray([0.0, 0.95, 0.16, 2.8, 0.99]), thresholds
    )
    assert low == (
        "Its global pattern has low variability, smooth point-to-point changes, "
        "low structural complexity, and weak short-term persistence."
    )
    assert mid == (
        "Its global pattern has moderate variability, moderate point-to-point "
        "changes, moderate structural complexity, and moderate short-term "
        "persistence."
    )
    assert high == (
        "Its global pattern has fairly high variability, fairly brisk point-to-point changes, "
        "fairly pronounced structural complexity, and strong short-term persistence."
    )


def test_trend_level_boundaries_match_five_classes():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    assert trend_level(-3.0, bounds) == "falls steeply"
    assert trend_level(-2.0, bounds) == "falls gradually"
    assert trend_level(-0.5, bounds) == "is mostly flat"
    assert trend_level(0.5, bounds) == "is mostly flat"
    assert trend_level(2.0, bounds) == "rises gradually"
    assert trend_level(3.0, bounds) == "rises steeply"


def test_rewrite_is_idempotent_and_updates_meta(tmp_path):
    rng = np.random.default_rng(42)
    attrs_rows = [
        [1, 0, 2],
        [0, 3, 1],
        [2, 1, 0],
        [1, 2, 3],
        [0, 0, 0],
    ]
    for split in SPLITS:
        curves = []
        for attrs in attrs_rows:
            background = rng.normal(0.0, 1.0, 128)
            ts, _ = _make_curve(attrs, rng, background=background)
            curves.append(ts)
        np.save(tmp_path / f"{split}_ts.npy", np.asarray(curves)[..., None].astype(np.float32))
        np.save(tmp_path / f"{split}_attrs_idx.npy", np.asarray(attrs_rows, dtype=np.int64))
        np.save(tmp_path / f"{split}_text_caps.npy", np.asarray([["obsolete"]] * 5))
    (tmp_path / "meta.json").write_text(
        json.dumps({"tsfresh_global": {}, "version": 3}), encoding="utf-8"
    )
    (tmp_path / "DATASET_CARD.md").write_text(
        "## synth-u style captions\nold content\n", encoding="utf-8"
    )

    summary = rewrite(tmp_path, n_jobs=1)
    first = np.load(tmp_path / "train_text_caps.npy")
    rewrite(tmp_path, n_jobs=1)

    assert "trend_bounds" in summary
    assert "pattern_thresholds" in summary
    np.testing.assert_array_equal(first, np.load(tmp_path / "train_text_caps.npy"))
    captions = np.load(tmp_path / "valid_text_caps.npy").reshape(-1)
    assert all("nothing" not in caption.lower() for caption in captions)
    assert all(caption.endswith("persistence.") for caption in captions)
    assert all("global pattern" in caption for caption in captions)
    # morphology sentence precedes the global part
    assert captions[0].index("beginning") < captions[0].index("global pattern")
    assert parse_segment_shapes(captions[0]) == ("single peak", "nothing", "double peaks")
    # previous captions were backed up before overwrite
    assert np.load(tmp_path / "train_text_caps.prev_synthustyle.npy").reshape(-1)[0] == "obsolete"
    # background features were saved
    assert np.load(tmp_path / "train_tsfresh_global_bg.npy").shape == (5, 5)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["caption_style"]["style"] == "synth-u-full"
    assert meta["caption_style"]["nothing_stages_silent"] is True
    assert meta["caption_style"]["global_features_computed_on"] == "pure background (ts - residual)"
    card = (tmp_path / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert card.count("## synth-u full captions") == 1
