"""Rewrite electricity_v3 captions: morphology-first + full global description.

Synth-u caption structure (verified against datasets/synth-u):
  morphology sentences (beginning/middle/end shapes)
  + global sentences (trend type & direction, season cycles, high-frequency period)

Electricity_v3 equivalent:
  - Morphology sentences FIRST, natural phrasing, ``nothing`` stages silent
    (kept from the synth-u style rewrite; this is what lifted segment hit rate
    from 50% to 88.9%).
  - Global description APPENDED AFTER the morphology sentences, two parts:
      1. coarse five-class trend sentence (slope buckets)
      2. global-pattern sentence (variability / point-to-point changes /
         structural complexity / short-term persistence, tertile buckets)

Global features are computed on the PURE BACKGROUND (ts - residual), not on the
final curve. The residual is rebuilt deterministically from ``attrs_idx`` and
``InjectionConfig``, so the global description is background-only and
independent of the morphology sentences (matching synth-u, where global
sentences describe background generation parameters).

Idempotent: captions are rebuilt from ``attrs_idx`` + background features
(never from previous caption text). Backs up the previous caption file
(``.prev_synthustyle.npy``) and updates ``meta.json`` / ``DATASET_CARD.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.data.datasets.semisynth_morph import (
    MORPHOLOGY_NAMES,
    STAGE_BOUNDS,
    STAGE_NAMES,
    InjectionConfig,
    morphology_residual,
)
from contsg.data.datasets.tsfresh_global import (
    FEATURE_NAMES,
    extract_global_features,
    fit_caption_thresholds,
)

SPLITS = ("train", "valid", "test")
DEFAULT_DATASET = ROOT / "datasets" / "electricity_15min_semisynth_morph"
BACKUP_SUFFIX = ".prev_synthustyle.npy"

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
    "sag": (
        "The {stage} part has a sag.",
        "There is a sag at the {stage} area.",
        "A sag at the {stage} area.",
    ),
}

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

DATASET_CARD_SECTION = """

## synth-u full captions (morphology + global background description)

`{split}_text_caps.npy` 是唯一 caption 版本：形态句前置、无形态（nothing）段沉默不写，
随后追加两部分全局描述——五类粗趋势句（`The series rises steeply overall.`，阈值用
train split 背景斜率 P10/P30/P70/P90 分位数）和全局 pattern 句（variability /
point-to-point changes / structural complexity / short-term persistence 四维
tertile 档位；最高档措辞已软化：fairly high / fairly brisk / fairly pronounced，
用于抑制模型对三连 high 档位的振荡过冲）。全局特征在**纯背景**（`ts - residual`）上
计算，与形态句信息独立。形态句式在三种 Synth-U 自然措辞之间确定性轮换。使用默认
`caption_variant="base"` 加载。
"""


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


def rebuild_residual(attrs: np.ndarray) -> np.ndarray:
    """Deterministically rebuild the injected morphology residual from attrs."""
    attrs = np.asarray(attrs, dtype=np.int64)
    if attrs.shape != (3,):
        raise ValueError(f"expected attrs with shape (3,), got {attrs.shape}")
    residual = np.zeros(128, dtype=np.float64)
    for attr, (start, stop) in zip(attrs, STAGE_BOUNDS):
        name = MORPHOLOGY_NAMES[int(attr)]
        residual[start:stop] += morphology_residual(
            name, stop - start, InjectionConfig()
        )
    return residual


def extract_background(curve: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    """Strip the injected residual to recover the normalized background."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (128,):
        raise ValueError(f"expected curve with shape (128,), got {curve.shape}")
    return curve - rebuild_residual(attrs)


def _level(value: float, bounds: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if value < bounds[0]:
        return labels[0]
    if value > bounds[1]:
        return labels[2]
    return labels[1]


def make_pattern_sentence(
    features: np.ndarray, thresholds: dict[str, tuple[float, float]]
) -> str:
    """One global-pattern sentence from four tertile-bucketed background features."""
    row = np.asarray(features, dtype=np.float64)
    if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
        raise ValueError(f"expected one finite feature row with shape ({len(FEATURE_NAMES)},)")
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
        name = MORPHOLOGY_NAMES[int(attr)]
        if name == "nothing":
            continue
        phrases = MORPHOLOGY_PHRASES[name]
        sentences.append(
            phrases[(sample_index + stage_index) % len(phrases)].format(stage=stage)
        )
    sentences.append(f"The series {trend_level(float(slope), trend_bounds)} overall.")
    sentences.append(make_pattern_sentence(pattern_features, pattern_thresholds))
    return " ".join(sentences)


def rewrite(dataset: Path, n_jobs: int = 0) -> dict:
    curves_by_split: dict[str, np.ndarray] = {}
    attrs_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        ts_path = dataset / f"{split}_ts.npy"
        attrs_path = dataset / f"{split}_attrs_idx.npy"
        if not ts_path.exists() or not attrs_path.exists():
            raise FileNotFoundError(f"missing {ts_path.name} or {attrs_path.name}")
        curves = np.load(ts_path)[..., 0]
        attrs = np.load(attrs_path)
        if attrs.shape != (len(curves), 3):
            raise ValueError(f"{split} attrs/ts row mismatch")
        curves_by_split[split] = curves
        attrs_by_split[split] = attrs

    # 1. Recover pure backgrounds and compute global features on them.
    background_features: dict[str, np.ndarray] = {}
    for split in SPLITS:
        backgrounds = np.asarray([
            extract_background(curve, attrs)
            for curve, attrs in zip(curves_by_split[split], attrs_by_split[split])
        ])
        background_features[split] = extract_global_features(backgrounds, n_jobs=n_jobs)

    # 2. Fit thresholds on train only.
    trend_bounds = fit_trend_bounds(background_features["train"][:, 0])
    pattern_thresholds = fit_caption_thresholds(background_features["train"])

    # 3. Write captions (idempotent; backed up once).
    lengths: dict[str, list[int]] = {}
    for split in SPLITS:
        attrs = attrs_by_split[split]
        slopes = background_features[split][:, 0]
        captions = np.empty((len(attrs), 1), dtype="U400")
        split_lengths = []
        for sample_index, (row, slope, feats) in enumerate(
            zip(attrs, slopes, background_features[split])
        ):
            captions[sample_index, 0] = make_caption(
                row, float(slope), trend_bounds, feats, pattern_thresholds, sample_index
            )
            split_lengths.append(len(captions[sample_index, 0].split()))
        lengths[split] = split_lengths

        caption_path = dataset / f"{split}_text_caps.npy"
        backup_path = dataset / f"{split}_text_caps{BACKUP_SUFFIX}"
        if caption_path.exists() and not backup_path.exists():
            shutil.copy2(caption_path, backup_path)
        np.save(caption_path, captions)
        np.save(dataset / f"{split}_tsfresh_global_bg.npy", background_features[split])

    # 4. Update metadata and dataset card.
    meta_path = dataset / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["caption_style"] = {
        "style": "synth-u-full",
        "morphology_first": True,
        "nothing_stages_silent": True,
        "global_trend_sentence": "coarse five-class trend sentence",
        "global_pattern_sentence": "variability / point-to-point changes / "
        "structural complexity / short-term persistence (tertiles)",
        "global_features_computed_on": "pure background (ts - residual)",
        "trend_bounds": list(trend_bounds),
        "trend_bounds_fit_split": "train",
        "trend_bounds_method": "background slope quantiles (0.10, 0.30, 0.70, 0.90)",
        "pattern_thresholds": {
            k: [float(v[0]), float(v[1])] for k, v in pattern_thresholds.items()
        },
        "pattern_threshold_fit_split": "train",
        "high_bucket_wording": "softened to fairly high / fairly brisk / "
        "fairly pronounced (v2, to curb oscillation overshoot)",
        "phrase_variants_per_shape": 3,
        "caption_file": "{split}_text_caps.npy",
        "background_feature_file": "{split}_tsfresh_global_bg.npy",
        "caption_variant": "base",
        "backup_suffix": BACKUP_SUFFIX,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_card_path = dataset / "DATASET_CARD.md"
    if dataset_card_path.exists():
        card = dataset_card_path.read_text(encoding="utf-8")
        if "## synth-u full captions" in card:
            pass  # already rewritten; keep idempotent
        else:
            card = card.rstrip() + "\n" + DATASET_CARD_SECTION
            dataset_card_path.write_text(card, encoding="utf-8")

    return {
        "trend_bounds": list(trend_bounds),
        "pattern_thresholds": {
            k: [float(v[0]), float(v[1])] for k, v in pattern_thresholds.items()
        },
        "mean_words_per_split": {
            split: float(np.mean(lengths[split])) for split in SPLITS
        },
        "backups_written": True,
    }


def main() -> None:
    args = parse_args()
    summary = rewrite(args.dataset.resolve(), n_jobs=args.n_jobs)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.dataset.resolve()}")


if __name__ == "__main__":
    main()
