#!/usr/bin/env python3
"""Visualize pure VerbalTS generations for all 12 stage-shape concepts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.provenance import sha256_file
from sae.shapes import SHAPE_NAMES, STAGE_NAMES, classify_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    return parser.parse_args()


def select_isolated_concepts(attrs: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Select one row per stage-shape pair, holding other stages at nothing."""
    selected: list[int] = []
    rows: list[dict[str, object]] = []
    for shape_index, shape in enumerate(SHAPE_NAMES):
        for stage_index, stage in enumerate(STAGE_NAMES):
            target = np.zeros(3, dtype=np.int64)
            target[stage_index] = shape_index
            matches = np.flatnonzero(np.all(attrs == target, axis=1))
            if not len(matches):
                raise ValueError(f"no validation row for isolated target {target.tolist()}")
            index = int(matches[0])
            selected.append(index)
            rows.append({
                "validation_index": index,
                "concept": f"{stage.lower()}_{shape.replace(' ', '_')}",
                "target_stage": stage.lower(),
                "target_shape": shape,
                "attrs_idx": target.tolist(),
            })
    return np.asarray(selected), rows


def add_boundaries(axis: plt.Axes) -> None:
    axis.axvline(42.5, color="0.55", linestyle=":", linewidth=.8)
    axis.axvline(85.5, color="0.55", linestyle=":", linewidth=.8)
    axis.grid(alpha=.15, linewidth=.5)


def save_plots(
    output_dir: Path,
    truth: np.ndarray,
    generated: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    x = np.arange(generated.shape[1])
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for axis, reference, sample, row in zip(axes.flat, truth, generated, rows):
        axis.plot(x, reference, color="0.55", linewidth=1.0, alpha=.8, label="reference")
        axis.plot(x, sample, color="#1565c0", linewidth=1.25, label="VerbalTS")
        add_boundaries(axis)
        conditioned = "/".join(SHAPE_NAMES[index] for index in row["attrs_idx"])
        predicted = "/".join(row["predicted_shapes"])
        axis.set_title(
            f"Condition (B/M/E): {conditioned}\nCNN pred. (B/M/E): {predicted}",
            fontsize=8,
        )
    axes[0, 0].legend(loc="upper left", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("timestep")
    for axis in axes[:, 0]:
        axis.set_ylabel("value")
    fig.suptitle(
        "VerbalTS v3: isolated stage-shape conditions\n"
        "B/M/E = beginning/middle/end; dotted lines are stage boundaries",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir / "concept_grid.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    for axis, sample, row in zip(axes.flat, generated, rows):
        axis.plot(x, sample, color="#6a1b9a", linewidth=1.3)
        add_boundaries(axis)
        axis.set_title(f"{row['concept']}\nattrs={row['attrs_idx']}", fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("timestep")
    fig.suptitle("VerbalTS v3 generated samples (one fixed-seed draw per condition)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir / "generated_gallery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    attrs = np.load(args.data_root / "valid_attrs_idx.npy")
    captions = np.load(args.data_root / "valid_text_caps.npy", allow_pickle=True).reshape(-1)
    embeddings = np.load(args.data_root / "valid_cap_emb.npy").astype(np.float32)
    truth_all = np.load(args.data_root / "valid_ts.npy")[:, :, 0]
    indices, rows = select_isolated_concepts(attrs)

    model, model_state_sha256 = load_verbalts(args.checkpoint, device)
    condition = torch.from_numpy(embeddings[indices]).to(device)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        generated = model.generate(condition, n_samples=1, sampler=args.sampler)[0, :, :, 0]
    generated_np = generated.cpu().float().numpy()

    classifier = PeakValleyClassifier1D(segment_len=43).to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_checkpoint, map_location=device, weights_only=True)
    )
    predictions = classify_curves(classifier, generated_np, device)
    for row, index, prediction in zip(rows, indices, predictions):
        row["caption"] = str(captions[index])
        row["predicted_shapes"] = [stage["shape"] for stage in prediction]
        row["stage_confidences"] = [stage["joint_confidence"] for stage in prediction]

    np.save(output_dir / "generated.npy", generated_np)
    np.save(output_dir / "reference.npy", truth_all[indices])
    save_plots(output_dir, truth_all[indices], generated_np, rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_state_sha256": model_state_sha256,
        "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
        "seed": args.seed,
        "sampler": args.sampler,
        "device": str(device),
        "rows": rows,
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()