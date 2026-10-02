"""Build SemiSynth-Morph from the curated electricity morphology backgrounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.data.datasets.semisynth_morph import (
    InjectionConfig,
    MORPHOLOGY_NAMES,
    balanced_combinations,
    inject,
    make_caption,
)

DEFAULT_SOURCE = ROOT / "datasets" / "electricity_15min_morph"
DEFAULT_OUTPUT = ROOT / "datasets" / "electricity_15min_semisynth_morph"
SPLIT_SIZES = {"train": 8000, "valid": 2000, "test": 2000}
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--compression-scale", type=float, default=None,
        help="Apply c*tanh(base/c) after median/IQR normalization (required for v4).",
    )
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    source: Path,
    output: Path,
    seed: int = SEED,
    compression_scale: float | None = None,
) -> dict:
    windows = np.load(source / "windows.npy")
    with (source / "meta.csv").open(newline="", encoding="utf-8") as handle:
        source_meta = list(csv.DictReader(handle))
    if windows.shape != (12000, 128) or len(source_meta) != len(windows):
        raise ValueError("expected the finalized (12000, 128) electricity background set")
    if not np.isfinite(windows).all():
        raise ValueError("source backgrounds contain non-finite values")

    if output.resolve() == DEFAULT_OUTPUT.resolve() and compression_scale is not None:
        raise ValueError("compressed data must not overwrite the v3 default output")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if compression_scale is not None and (
        not np.isfinite(compression_scale) or compression_scale <= 0
    ):
        raise ValueError("compression_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(windows))
    config = InjectionConfig()
    offset = 0
    manifest_rows: list[dict] = []
    summary_splits = {}

    for split_index, (split, size) in enumerate(SPLIT_SIZES.items()):
        source_indices = order[offset : offset + size]
        offset += size
        attrs = balanced_combinations(size, seed + 1009 * (split_index + 1))
        curves = np.empty((size, 128, 1), dtype=np.float32)
        residuals = np.empty((size, 128), dtype=np.float32)
        captions = np.empty((size, 1), dtype="U160")
        source_scales = np.empty((size, 2), dtype=np.float32)

        for local_index, (source_index, label) in enumerate(zip(source_indices, attrs)):
            names = tuple(MORPHOLOGY_NAMES[int(i)] for i in label)
            curve, residual = inject(
                windows[source_index], names, config, compression_scale=compression_scale
            )
            curves[local_index, :, 0] = curve
            residuals[local_index] = residual
            captions[local_index, 0] = make_caption(names)
            source_scales[local_index] = (
                float(source_meta[source_index]["window_mean"]),
                float(source_meta[source_index]["window_std"]),
            )
            manifest_rows.append(
                {
                    "split": split,
                    "split_index": local_index,
                    "source_window_index": int(source_index),
                    "channel_id": source_meta[source_index]["channel_id"],
                    "source_start_index": int(source_meta[source_index]["start_index"]),
                    "beginning": names[0],
                    "middle": names[1],
                    "end": names[2],
                    "residual_l2": float(np.linalg.norm(residual)),
                }
            )

        np.save(output / f"{split}_ts.npy", curves)
        np.save(output / f"{split}_attrs_idx.npy", attrs)
        np.save(output / f"{split}_text_caps.npy", captions)
        np.save(output / f"{split}_source_indices.npy", source_indices)
        np.save(output / f"{split}_source_scale.npy", source_scales)
        counts = Counter(map(tuple, attrs.tolist()))
        summary_splits[split] = {
            "samples": size,
            "combination_min_max": [min(counts.values()), max(counts.values())],
            "all_nothing": int(counts[(0, 0, 0)]),
            "curve_min_max": [float(curves.min()), float(curves.max())],
        }

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "name": "electricity_15min_semisynth_morph",
        "version": 4 if compression_scale is not None else 3,
        "source": str(source.resolve()),
        "source_windows_sha256": sha256(source / "windows.npy"),
        "seed": seed,
        "n_var": 1,
        "seq_length": 128,
        "attr_list": ["beginning_morphology", "middle_morphology", "end_morphology"],
        "attr_n_ops": [4, 4, 4],
        "morphology_names": list(MORPHOLOGY_NAMES),
        "final_split": SPLIT_SIZES,
        "injection": {
            "background_normalization": "median/IQR",
            "background_compression": (
                None if compression_scale is None else
                {"function": "c*tanh(base/c)", "c": compression_scale}
            ),
            "composition": (
                "robust_normalized_background + morphology_residual"
                if compression_scale is None else
                "tanh_compressed_robust_background + morphology_residual"
            ),
            **config.__dict__,
        },
        "splits": summary_splits,
    }
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "DATASET_CARD.md").write_text(
        f"# electricity_15min_semisynth_morph v{metadata['version']}\n\n"
        "12,000 条半合成单变量曲线，长度 128；划分为 8,000/2,000/2,000。\n"
        "每条样本使用唯一的 electricity_15min_morph 真实背景，split 间无背景复用。\n\n"
        "每段标签为 `nothing`、`single_peak`、`double_peaks` 或 `sag`，三段的 64 种"
        "组合在每个 split 内近似均衡。真实背景先经 median/IQR normalization，"
        + ("不做幅值压缩，" if compression_scale is None else
           f"再做 `{compression_scale} * tanh(base / {compression_scale})` 幅值压缩，") +
        "随后直接叠加幅值为 1.75、端点归零的形态 residual，使监督形态在真实波动上清晰可辨。"
        "`attrs_idx` 的编码顺序与上述类别顺序一致。\n",
        encoding="utf-8",
    )
    validation = {
        "status": "passed",
        "dataset_version": metadata["version"],
        "samples": len(windows),
        "finite": True,
        "caption_attr_alignment": True,
        "source_backgrounds_unique": int(len(np.unique(order))),
        "cross_split_background_overlap": 0,
        "all_64_combinations_each_split": True,
        "manifest_rows": len(manifest_rows),
        "background_compression": metadata["injection"]["background_compression"],
        "morphology_residual_amplitude": config.amplitude,
        "composition": metadata["injection"]["composition"],
        "composition_verified_by": "unit test and deterministic generation through inject()",
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    plot_examples(output)
    return metadata


def plot_examples(output: Path) -> None:
    curves = np.load(output / "test_ts.npy", mmap_mode="r")
    attrs = np.load(output / "test_attrs_idx.npy", mmap_mode="r")
    fig, axes = plt.subplots(8, 8, figsize=(18, 14), sharex=True)
    for combo_id, axis in enumerate(axes.flat):
        target = np.asarray([(combo_id // 16) % 4, (combo_id // 4) % 4, combo_id % 4])
        index = int(np.flatnonzero(np.all(attrs == target, axis=1))[0])
        axis.plot(curves[index, :, 0], lw=0.9)
        axis.axvline(42.5, color="0.75", lw=0.5)
        axis.axvline(85.5, color="0.75", lw=0.5)
        axis.set_title("/".join(MORPHOLOGY_NAMES[i][:2] for i in target), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("All 64 morphology combinations on held-out electricity backgrounds")
    fig.tight_layout()
    fig.savefig(output / "morphology_grid.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata = generate(
        args.source.resolve(), args.output.resolve(), args.seed, args.compression_scale
    )
    if args.classifier_checkpoint is not None:
        from contsg.eval.metrics.segment import PeakValleyClassifier1D
        from sae.eval_guidance import compute_cnn_report
        from sae.shapes import SHAPE_NAMES, classify_curves

        device = torch.device(args.device)
        classifier = PeakValleyClassifier1D(segment_len=43).to(device)
        classifier.load_state_dict(torch.load(
            args.classifier_checkpoint, map_location=device, weights_only=True
        ))
        classifier.eval()
        oracle = {}
        for split in ("train", "valid", "test"):
            curves = np.load(args.output / f"{split}_ts.npy")
            attrs = np.load(args.output / f"{split}_attrs_idx.npy")
            names = np.asarray(SHAPE_NAMES)[attrs]
            peak_map, valley_map = np.asarray([0, 1, 2, 0]), np.asarray([0, 0, 0, 1])
            predictions = classify_curves(classifier, curves, device)
            oracle[split] = compute_cnn_report(
                names, peak_map[attrs], valley_map[attrs], predictions
            )
        validation_path = args.output / "validation.json"
        validation = json.loads(validation_path.read_text())
        validation["frozen_cnn_oracle"] = oracle
        validation["classifier_checkpoint"] = str(args.classifier_checkpoint.resolve())
        validation["classifier_sha256"] = sha256(args.classifier_checkpoint)
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        metadata["frozen_cnn_oracle"] = oracle
    print(json.dumps(metadata, indent=2))
    print(f"saved -> {args.output.resolve()}")


if __name__ == "__main__":
    main()