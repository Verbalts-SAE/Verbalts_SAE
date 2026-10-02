"""Rewrite electricity_v3 captions in synth-u style.

Synth-u caption construction rules (verified against datasets/synth-u):
- Morphology sentences come FIRST and use natural phrasing, e.g.
  "The beginning part has a single peak." / "There are double peaks at the
  middle area." / "A sag at the end area."
- Stages with shape ``nothing`` are SILENT: no sentence is written for them.
- Global information is a single coarse trend sentence appended AFTER the
  morphology sentences (synth-u uses trend/season sentences at random
  positions; we fix morphology-first to protect shape semantics).

The old caption format (tsfresh five-feature global summary FIRST + explicit
"The X part has nothing." sentences) diluted shape semantics and used weak
negation phrasing; both are removed here.

This script is idempotent: it rebuilds captions from ``attrs_idx`` and
``tsfresh_global`` (never from the previous caption text), backs up the
previous caption file, and updates ``meta.json`` / ``DATASET_CARD.md``.
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

from contsg.data.datasets.semisynth_morph import MORPHOLOGY_NAMES, STAGE_NAMES

SPLITS = ("train", "valid", "test")
DEFAULT_DATASET = ROOT / "datasets" / "electricity_15min_semisynth_morph"
BACKUP_SUFFIX = ".prev_tsfresh_full.npy"

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

DATASET_CARD_SECTION = """

## synth-u style captions

`{split}_text_caps.npy` 是唯一 caption 版本，严格对齐 Synth-U 的构造规则：
形态句前置、无形态（nothing）段沉默不写、全局信息降级为后置的一句粗分类趋势句
（`The series rises steeply overall.` 等五类，阈值仅用 train split 的 1/10、3/10、
7/10、9/10 斜率分位数拟合并写入 `meta.json`）。形态句式在三种 Synth-U 自然措辞
之间确定性轮换（如 `The beginning part has a single peak.` / `There is a single
peak at the beginning area.` / `A single peak at the beginning area.`）。使用默认
`caption_variant="base"` 加载。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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


def make_caption(
    attrs: np.ndarray,
    slope: float,
    bounds: tuple[float, float, float, float],
    sample_index: int,
) -> str:
    """Build one synth-u style caption from an (3,) attrs row and trend slope."""
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
    sentences.append(f"The series {trend_level(float(slope), bounds)} overall.")
    return " ".join(sentences)


def rewrite(dataset: Path) -> dict:
    attrs_by_split: dict[str, np.ndarray] = {}
    features_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        attrs_path = dataset / f"{split}_attrs_idx.npy"
        features_path = dataset / f"{split}_tsfresh_global.npy"
        if not attrs_path.exists() or not features_path.exists():
            raise FileNotFoundError(
                f"missing {attrs_path.name} or {features_path.name} in {dataset}"
            )
        attrs_by_split[split] = np.load(attrs_path)
        features_by_split[split] = np.load(features_path)

    bounds = fit_trend_bounds(features_by_split["train"][:, 0])

    lengths: dict[str, list[int]] = {}
    trend_counts = np.zeros(len(TREND_LABELS), dtype=np.int64)
    silent_samples = {split: 0 for split in SPLITS}
    for split in SPLITS:
        attrs = attrs_by_split[split]
        slopes = features_by_split[split][:, 0]
        if attrs.shape != (len(slopes), 3):
            raise ValueError(f"{split} attrs/features row mismatch")
        captions = np.empty((len(attrs), 1), dtype="U160")
        split_lengths = []
        for sample_index, (row, slope) in enumerate(zip(attrs, slopes)):
            captions[sample_index, 0] = make_caption(
                row, float(slope), bounds, sample_index
            )
            split_lengths.append(len(captions[sample_index, 0].split()))
            trend_counts[TREND_LABELS.index(trend_level(float(slope), bounds))] += 1
            if not np.any(row != 0):
                silent_samples[split] += 1
        lengths[split] = split_lengths

        caption_path = dataset / f"{split}_text_caps.npy"
        backup_path = dataset / f"{split}_text_caps{BACKUP_SUFFIX}"
        if caption_path.exists() and not backup_path.exists():
            shutil.copy2(caption_path, backup_path)
        np.save(caption_path, captions)

    meta_path = dataset / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["caption_style"] = {
        "style": "synth-u",
        "morphology_first": True,
        "nothing_stages_silent": True,
        "global_trend_sentence": "single coarse five-class sentence appended last",
        "trend_bounds": list(bounds),
        "trend_bounds_fit_split": "train",
        "trend_bounds_method": "slope quantiles (0.10, 0.30, 0.70, 0.90)",
        "phrase_variants_per_shape": 3,
        "caption_file": "{split}_text_caps.npy",
        "caption_variant": "base",
        "backup_suffix": BACKUP_SUFFIX,
    }
    meta["tsfresh_global"]["caption_structure"] = (
        "morphology-first synth-u style; nothing stages silent; "
        "one coarse trend sentence appended last"
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_card_path = dataset / "DATASET_CARD.md"
    if dataset_card_path.exists():
        card = dataset_card_path.read_text(encoding="utf-8")
        if "## synth-u style captions" in card:
            pass  # already rewritten; keep idempotent
        else:
            heading = "## tsfresh global + fine-grained captions"
            if heading in card:
                before, section = card.split(heading, 1)
                next_heading = section.find("\n## ")
                suffix = section[next_heading:] if next_heading >= 0 else ""
                card = before.rstrip() + "\n" + DATASET_CARD_SECTION + suffix
            else:
                card = card.rstrip() + "\n" + DATASET_CARD_SECTION
            dataset_card_path.write_text(card, encoding="utf-8")

    return {
        "trend_bounds": list(bounds),
        "trend_class_counts": dict(zip(TREND_LABELS, trend_counts.tolist())),
        "mean_words_per_split": {
            split: float(np.mean(lengths[split])) for split in SPLITS
        },
        "all_nothing_samples": silent_samples,
        "backups_written": True,
    }


def main() -> None:
    args = parse_args()
    summary = rewrite(args.dataset.resolve())
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.dataset.resolve()}")


if __name__ == "__main__":
    main()
