"""Build etth2_2class captions in the synth-u full style (morphology + global).

Follows ``rewrite_electricity_v3_global_captions.py`` conventions:
- Morphology sentences FIRST with natural synth-u phrasing, three variants
  rotated deterministically.  ``nothing`` stages stay silent (etth2 has none).
- One coarse five-class trend sentence appended afterwards
  (``The series rises steeply overall.``), thresholds fitted on the train
  split slope quantiles (0.10 / 0.30 / 0.70 / 0.90).
- One global-pattern sentence (variability / point-to-point changes /
  structural complexity / short-term persistence), train-only tertiles with
  softened high buckets (fairly high / fairly brisk / fairly pronounced).
- No concrete numbers appear in captions.

Unlike the electricity semisynth dataset, ETTh2 windows are real data with no
injected residual, so global features are extracted on the curve itself and
stored as ``{split}_tsfresh_global.npy`` (the file ``eval_morph_steering``
expects for global-category metrics).

Additionally derives ``{split}_attrs_idx.npy`` in the four-way shape
vocabulary used by every downstream component:
``0/1 labels -> single peak (1) / double peaks (2)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.data.datasets.tsfresh_global import (
    FEATURE_NAMES,
    extract_global_features,
    fit_caption_thresholds,
)

SPLITS = ("train", "valid", "test")
DEFAULT_DATASET = ROOT / "datasets" / "etth2_2class" / "etth2_2class_dataset"

STAGE_NAMES = ("beginning", "middle", "end")

# Synth-u style phrasings per shape; three variants are rotated deterministically.
MORPHOLOGY_PHRASES = {
    "single_peak": (
        "The {stage} part has a single peak.",
        "There is a single peak at the {stage} area.",
        "A single peak at the {stage} area.",
    ),
    "double_peaks": (
        "The {stage} part has double peaks.",
        "There are double peaks at the {stage} area.",
        "Double peaks at the {stage} area.",
    ),
}

# etth2 label -> four-way shape vocabulary index (SHAPE_NAMES order).
LABEL_TO_SHAPE = {0: 1, 1: 2}  # single peak, double peaks

TREND_LABELS = (
    "falls steeply",
    "falls gradually",
    "is mostly flat",
    "rises gradually",
    "rises steeply",
)
TREND_QUANTILES = (0.10, 0.30, 0.70, 0.90)

VARIABILITY_LABELS = ("low", "moderate", "fairly high")
CHANGES_LABELS = ("smooth", "moderate", "fairly brisk")
COMPLEXITY_LABELS = ("low", "moderate", "fairly pronounced")
PERSISTENCE_LABELS = ("weak", "moderate", "strong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--n-jobs", type=int, default=0)
    return parser.parse_args()


def fit_trend_bounds(slopes: np.ndarray) -> tuple[float, float, float, float]:
    """Fit train-only slope bounds for the five coarse trend classes."""
    values = np.asarray(slopes, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("expected a non-empty finite 1-D slope array")
    return tuple(float(v) for v in np.quantile(values, TREND_QUANTILES))


def trend_level(slope: float, bounds: tuple[float, float, float, float]) -> str:
    if slope < bounds[0]:
        return TREND_LABELS[0]
    if slope < bounds[1]:
        return TREND_LABELS[1]
    if slope <= bounds[2]:
        return TREND_LABELS[2]
    if slope <= bounds[3]:
        return TREND_LABELS[3]
    return TREND_LABELS[4]


def _level(
    value: float, bounds: tuple[float, float], labels: tuple[str, str, str]
) -> str:
    if value < bounds[0]:
        return labels[0]
    if value > bounds[1]:
        return labels[2]
    return labels[1]


def make_pattern_sentence(
    features: np.ndarray, thresholds: dict[str, tuple[float, float]]
) -> str:
    """One global-pattern sentence from four tertile-bucketed features."""
    row = np.asarray(features, dtype=np.float64)
    if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
        raise ValueError(
            f"expected one finite feature row with shape ({len(FEATURE_NAMES)},)"
        )
    variability = _level(row[1], thresholds["standard_deviation"], VARIABILITY_LABELS)
    changes = _level(row[2], thresholds["mean_abs_change"], CHANGES_LABELS)
    complexity = _level(row[3], thresholds["cid_ce_normalized"], COMPLEXITY_LABELS)
    persistence = _level(
        row[4], thresholds["autocorrelation_lag_1"], PERSISTENCE_LABELS
    )
    return (
        f"Its global pattern has {variability} variability, {changes} point-to-point "
        f"changes, {complexity} structural complexity, and {persistence} short-term "
        f"persistence."
    )


def make_caption(
    attrs: np.ndarray,
    slope: float,
    trend_bounds: tuple[float, float, float, float],
    pattern_features: np.ndarray,
    pattern_thresholds: dict[str, tuple[float, float]],
    sample_index: int,
) -> str:
    """Build one caption: morphology sentences + trend sentence + pattern sentence."""
    attrs = np.asarray(attrs, dtype=np.int64)
    if attrs.shape != (3,):
        raise ValueError(f"expected attrs with shape (3,), got {attrs.shape}")
    sentences = []
    for stage_index, (stage, attr) in enumerate(zip(STAGE_NAMES, attrs)):
        # etth2 only carries single peak / double peaks (no 'nothing' rows).
        shape_index = int(attr)
        if shape_index == 0:
            continue
        name = ("single_peak", "double_peaks")[shape_index - 1]
        phrases = MORPHOLOGY_PHRASES[name]
        sentences.append(
            phrases[(sample_index + stage_index) % len(phrases)].format(stage=stage)
        )
    sentences.append(f"The series {trend_level(float(slope), trend_bounds)} overall.")
    sentences.append(make_pattern_sentence(pattern_features, pattern_thresholds))
    return " ".join(sentences)


def build(dataset: Path, n_jobs: int = 0) -> dict:
    curves_by_split: dict[str, np.ndarray] = {}
    attrs_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        ts_path = dataset / f"{split}_ts.npy"
        labels_path = dataset / f"{split}_labels.npy"
        if not ts_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"missing {ts_path.name} or {labels_path.name}")
        curves = np.load(ts_path)[..., 0]
        labels = np.load(labels_path)
        if labels.shape != (len(curves), 3):
            raise ValueError(f"{split} labels/ts row mismatch")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{split} labels must be binary (0/1)")
        attrs = np.asarray(
            [[LABEL_TO_SHAPE[int(value)] for value in row] for row in labels],
            dtype=np.int64,
        )
        curves_by_split[split] = curves
        attrs_by_split[split] = attrs
        np.save(dataset / f"{split}_attrs_idx.npy", attrs)

    # 1. Global features on the curve itself (real data: no injected residual).
    features_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features_by_split[split] = extract_global_features(
            curves_by_split[split], n_jobs=n_jobs
        )
        np.save(dataset / f"{split}_tsfresh_global.npy", features_by_split[split])

    # 2. Fit thresholds on train only.
    trend_bounds = fit_trend_bounds(features_by_split["train"][:, 0])
    pattern_thresholds = fit_caption_thresholds(features_by_split["train"])

    # 3. Write captions.
    lengths: dict[str, list[int]] = {}
    trend_counts = np.zeros(len(TREND_LABELS), dtype=np.int64)
    for split in SPLITS:
        attrs = attrs_by_split[split]
        slopes = features_by_split[split][:, 0]
        captions = np.empty((len(attrs), 1), dtype="U400")
        split_lengths = []
        for sample_index, (row, slope, feats) in enumerate(
            zip(attrs, slopes, features_by_split[split])
        ):
            captions[sample_index, 0] = make_caption(
                row, float(slope), trend_bounds, feats, pattern_thresholds,
                sample_index,
            )
            split_lengths.append(len(captions[sample_index, 0].split()))
            trend_counts[TREND_LABELS.index(trend_level(float(slope), trend_bounds))] += 1
        lengths[split] = split_lengths
        np.save(dataset / f"{split}_text_caps.npy", captions)

    # 4. Update metadata.
    meta_path = dataset / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["attrs_idx"] = {
        "derived_from": "labels.npy",
        "mapping": {"0": "single peak (shape id 1)", "1": "double peaks (shape id 2)"},
        "file": "{split}_attrs_idx.npy",
    }
    meta["caption_style"] = {
        "style": "synth-u-full",
        "morphology_first": True,
        "nothing_stages_silent": True,
        "global_trend_sentence": "coarse five-class trend sentence",
        "global_pattern_sentence": "variability / point-to-point changes / "
        "structural complexity / short-term persistence (tertiles)",
        "global_features_computed_on": "the curve itself (real ETTh2 windows)",
        "trend_bounds": list(trend_bounds),
        "trend_bounds_fit_split": "train",
        "trend_bounds_method": "slope quantiles (0.10, 0.30, 0.70, 0.90)",
        "pattern_thresholds": {
            key: [float(value[0]), float(value[1])]
            for key, value in pattern_thresholds.items()
        },
        "pattern_threshold_fit_split": "train",
        "high_bucket_wording": "softened to fairly high / fairly brisk / "
        "fairly pronounced (v2, to curb oscillation overshoot)",
        "phrase_variants_per_shape": 3,
        "caption_uses_numeric_values": False,
        "caption_file": "{split}_text_caps.npy",
        "global_feature_file": "{split}_tsfresh_global.npy",
        "caption_variant": "base",
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "trend_bounds": list(trend_bounds),
        "pattern_thresholds": {
            key: [float(value[0]), float(value[1])]
            for key, value in pattern_thresholds.items()
        },
        "trend_class_counts": dict(zip(TREND_LABELS, trend_counts.tolist())),
        "mean_words_per_split": {
            split: float(np.mean(lengths[split])) for split in SPLITS
        },
    }


def main() -> None:
    args = parse_args()
    summary = build(args.dataset.resolve(), n_jobs=args.n_jobs)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.dataset.resolve()}")


if __name__ == "__main__":
    main()
