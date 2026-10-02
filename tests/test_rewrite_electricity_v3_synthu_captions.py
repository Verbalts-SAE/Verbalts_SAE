import json

import numpy as np

from contsg.data.datasets.semisynth_morph import MORPHOLOGY_NAMES
from sae.shapes import parse_segment_shapes
from scripts.rewrite_electricity_v3_synthu_captions import (
    DATASET_CARD_SECTION,
    SPLITS,
    fit_trend_bounds,
    make_caption,
    rewrite,
    trend_level,
)


def test_make_caption_is_morphology_first_and_nothing_silent():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    caption = make_caption(np.asarray([3, 0, 1]), slope=0.9, bounds=bounds, sample_index=0)
    assert caption == (
        "The beginning part has a sag. A single peak at the end area. "
        "The series rises gradually overall."
    )
    assert "nothing" not in caption.lower()
    # Morphology sentence appears before the trend sentence.
    assert caption.index("sag") < caption.index("overall")


def test_make_caption_phrase_variants_rotate_deterministically():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    attrs = np.asarray([1, 1, 1])
    captions = [make_caption(attrs, 0.0, bounds, i) for i in range(3)]
    # Each variant starts with a different opener, and the cycle repeats.
    assert len({caption.split()[0] for caption in captions}) == 3
    assert make_caption(attrs, 0.0, bounds, 3) == captions[0]
    assert make_caption(attrs, 0.0, bounds, 4) == captions[1]


def test_make_caption_all_nothing_keeps_only_trend_sentence():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    caption = make_caption(np.asarray([0, 0, 0]), slope=-3.0, bounds=bounds, sample_index=0)
    assert caption == "The series falls steeply overall."
    assert parse_segment_shapes(caption) == ("nothing", "nothing", "nothing")


def test_make_caption_round_trips_through_parser():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    attrs = np.asarray([1, 2, 3])
    caption = make_caption(attrs, slope=0.0, bounds=bounds, sample_index=4)
    parsed = parse_segment_shapes(caption)
    expected = tuple(MORPHOLOGY_NAMES[int(v)].replace("_", " ") for v in attrs)
    assert parsed == expected


def test_make_caption_rejects_wrong_attr_shape():
    try:
        make_caption(np.asarray([1, 2]), 0.0, (-2.0, -0.5, 0.5, 2.0), 0)
    except ValueError as error:
        assert "shape (3,)" in str(error)
    else:
        raise AssertionError("expected wrong-shaped attrs to be rejected")


def test_trend_level_boundaries_match_five_classes():
    bounds = (-2.0, -0.5, 0.5, 2.0)
    assert trend_level(-3.0, bounds) == "falls steeply"
    assert trend_level(-2.0, bounds) == "falls gradually"
    assert trend_level(-0.5, bounds) == "is mostly flat"
    assert trend_level(0.5, bounds) == "is mostly flat"
    assert trend_level(2.0, bounds) == "rises gradually"
    assert trend_level(3.0, bounds) == "rises steeply"


def test_fit_trend_bounds_uses_quantiles_and_rejects_bad_input():
    slopes = np.linspace(-3.0, 3.0, 101)
    bounds = fit_trend_bounds(slopes)
    assert np.allclose(bounds, np.quantile(slopes, (0.10, 0.30, 0.70, 0.90)))
    for bad in (np.array([]), np.asarray([1.0, np.nan]), np.asarray([[1.0]])):
        try:
            fit_trend_bounds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected bad slopes to be rejected: {bad!r}")


def test_rewrite_is_idempotent_and_updates_meta(tmp_path):
    attrs = np.asarray(
        [
            [1, 0, 2],
            [0, 3, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    slopes = np.asarray([[-2.5], [0.0], [2.5]], dtype=np.float32)
    for split in SPLITS:
        np.save(tmp_path / f"{split}_attrs_idx.npy", attrs)
        np.save(tmp_path / f"{split}_tsfresh_global.npy", slopes)
        np.save(tmp_path / f"{split}_text_caps.npy", np.asarray([["obsolete"]] * 3))
    (tmp_path / "meta.json").write_text(
        json.dumps({"tsfresh_global": {}, "version": 3}), encoding="utf-8"
    )
    (tmp_path / "DATASET_CARD.md").write_text(
        "## tsfresh global + fine-grained captions\nold content\n", encoding="utf-8"
    )

    summary = rewrite(tmp_path)
    first = np.load(tmp_path / "train_text_caps.npy")
    rewrite(tmp_path)

    # np.quantile([-2.5, 0.0, 2.5], q) with n=3 gives positions 0.2/0.6/1.4/1.8.
    assert np.allclose(summary["trend_bounds"], [-2.0, -1.0, 1.0, 2.0])
    assert summary["trend_class_counts"] == {
        "falls steeply": 3, "falls gradually": 0, "is mostly flat": 3,
        "rises gradually": 0, "rises steeply": 3,
    }
    assert summary["all_nothing_samples"] == {split: 0 for split in SPLITS}
    np.testing.assert_array_equal(first, np.load(tmp_path / "train_text_caps.npy"))
    captions = np.load(tmp_path / "valid_text_caps.npy").reshape(-1)
    assert all("nothing" not in caption.lower() for caption in captions)
    assert all(caption.endswith("overall.") for caption in captions)
    assert parse_segment_shapes(captions[0]) == ("single peak", "nothing", "double peaks")
    # Previous captions were backed up before overwrite.
    assert np.load(tmp_path / "train_text_caps.prev_tsfresh_full.npy").reshape(-1)[0] == "obsolete"
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["caption_style"]["style"] == "synth-u"
    assert meta["caption_style"]["nothing_stages_silent"] is True
    card = (tmp_path / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert card.count("## synth-u style captions") == 1
    assert "old content" not in card
    assert DATASET_CARD_SECTION.strip() in card
