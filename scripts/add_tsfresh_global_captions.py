"""Write global+fine-grained descriptions as the sole canonical captions."""

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
    CAPTION_FEATURE_NAMES,
    FEATURE_NAMES,
    combine_global_and_local_caption,
    extract_global_features,
    fit_caption_thresholds,
    make_global_description,
)


DEFAULT_DATASET = ROOT / "datasets" / "electricity_15min_semisynth_morph"
SPLITS = ("train", "valid", "test")
DATASET_CARD_SECTION = """

## tsfresh global + fine-grained captions

`{split}_tsfresh_global.npy` 保存五个全局特征：线性趋势斜率、标准差、平均绝对变化、
归一化 CID complexity 和 lag-1 autocorrelation。离散描述的阈值仅使用 train split 的
1/3 与 2/3 分位数拟合，并写入 `meta.json`。

`{split}_text_caps.npy` 是唯一 caption 版本，先将五个数值特征分别转成不含具体数字的类别描述：
整体趋势方向、全局波动、点间变化、结构复杂度和短期持续性；随后明确引出原有
beginning/middle/end 细粒度 morphology。使用默认 `caption_variant="base"` 加载，
不再保留纯局部描述或单独的 tsfresh caption 文件。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--n-jobs", type=int, default=0)
    return parser.parse_args()


def augment(dataset: Path, n_jobs: int = 0) -> dict:
    """Replace local captions with canonical combined captions, idempotently."""

    feature_matrices: dict[str, np.ndarray] = {}
    local_captions: dict[str, np.ndarray] = {}
    for split in SPLITS:
        curves = np.load(dataset / f"{split}_ts.npy", mmap_mode="r")
        caption_path = dataset / f"{split}_caps.npy"
        if not caption_path.exists():
            caption_path = dataset / f"{split}_text_caps.npy"
        if not caption_path.exists():
            raise FileNotFoundError(f"no base caption file found for {split} in {dataset}")
        captions = np.load(caption_path, allow_pickle=True).reshape(-1)
        # On subsequent runs the canonical input already has a global summary.
        # Keep its original local sentences rather than prepending another summary.
        locals_only = []
        for caption in captions:
            text = str(caption)
            if text.startswith("The time series has "):
                start = text.find("The beginning part has ")
                if start < 0:
                    raise ValueError(f"cannot locate local morphology in {split} caption")
                text = text[start:]
            locals_only.append(text)
        captions = np.asarray(locals_only)
        if len(curves) != len(captions):
            raise ValueError(f"{split} curves and captions have different sample counts")
        feature_matrices[split] = extract_global_features(curves, n_jobs=n_jobs)
        local_captions[split] = captions

    thresholds = fit_caption_thresholds(feature_matrices["train"])
    max_caption_length = 0
    for split in SPLITS:
        features = feature_matrices[split]
        combined = np.asarray([
            combine_global_and_local_caption(
                make_global_description(row, thresholds), str(local_caption)
            ) for row, local_caption in zip(features, local_captions[split])
        ])
        max_caption_length = max(max_caption_length, max(map(len, combined), default=0))
        np.save(dataset / f"{split}_tsfresh_global.npy", features)
        np.save(dataset / f"{split}_text_caps.npy", combined)
        for obsolete in (f"{split}_caps.npy", f"{split}_text_caps_tsfresh.npy"):
            (dataset / obsolete).unlink(missing_ok=True)

    metadata_path = dataset / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tsfresh_global"] = {
        "tsfresh_version": __import__("tsfresh").__version__,
        "feature_names": list(FEATURE_NAMES),
        "caption_feature_names": list(CAPTION_FEATURE_NAMES),
        "caption_structure": (
            "categorical five-feature global summary followed by fine-grained morphology"
        ),
        "caption_uses_numeric_values": False,
        "feature_file": "{split}_tsfresh_global.npy",
        "caption_variant": "base",
        "caption_file": "{split}_text_caps.npy",
        "threshold_fit_split": "train",
        "threshold_method": "tertiles (1/3 and 2/3 quantiles)",
        "thresholds": {name: list(bounds) for name, bounds in thresholds.items()},
        "max_caption_length": max_caption_length,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validation_path = dataset / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["caption_contains_morphology"] = True
        validation.pop("morphology_condition_file", None)
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    dataset_card_path = dataset / "DATASET_CARD.md"
    if dataset_card_path.exists():
        dataset_card = dataset_card_path.read_text(encoding="utf-8")
        heading = "## tsfresh global + fine-grained captions"
        if heading in dataset_card:
            before, section = dataset_card.split(heading, 1)
            next_heading = section.find("\n## ")
            suffix = section[next_heading:] if next_heading >= 0 else ""
            dataset_card = before.rstrip() + "\n" + DATASET_CARD_SECTION + suffix
        else:
            dataset_card = dataset_card.rstrip() + "\n" + DATASET_CARD_SECTION
        dataset_card_path.write_text(dataset_card, encoding="utf-8")
    return metadata["tsfresh_global"]


def main() -> None:
    args = parse_args()
    summary = augment(args.dataset.resolve(), args.n_jobs)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.dataset.resolve()}")


if __name__ == "__main__":
    main()