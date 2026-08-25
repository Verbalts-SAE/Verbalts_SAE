"""Reconstruction comparison figures: Pure VerbalTS vs the t=5-45 Top-K SAE on synth-u.

The figure for each test case overlays the two generated curves and shows the
caption-derived shape ground truth plus the CNN prediction for each generated
curve's beginning/middle/end segments.

The segment CNN is trained directly from the retained synth-u arrays.  Its
targets are extracted from the local shape statements in each caption; an
unmentioned segment is labelled ``nothing``.  Derived classifier weights and
figures are written to the output directory, never to ``artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from contsg.config.schema import ExperimentConfig
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from contsg.registry import Registry
from sae.activations import load_verbalts
from sae.shapes import (
    SHAPE_NAMES,
    SHAPE_TO_TARGET,
    STAGE_NAMES,
    captions_to_targets,
    classify_curves,
    parse_segment_shapes,
    split_segments,
)
from sae.steering import attach_sae, detach_sae

# Import for the Registry side effect.
import contsg.models.verbalts  # noqa: F401, E402


LOGGER = logging.getLogger(__name__)

VARIANTS = (
    ("Pure VerbalTS", None, "#0072B2"),
    ("Top-K SAE (t=5–45)", "results/sae_retrain/models/t5_45/best.pt", "#009E73"),
)


def _load_split_dataset(data_root: Path, split: str) -> tuple[TensorDataset, np.ndarray]:
    curves = np.load(data_root / f"{split}_ts.npy", mmap_mode="r")
    captions = np.load(data_root / f"{split}_text_caps.npy")
    shape_names, peak, valley = captions_to_targets(captions)
    segments = split_segments(curves).reshape(-1, 43).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(segments[:, None, :]),
        torch.from_numpy(peak.reshape(-1)),
        torch.from_numpy(valley.reshape(-1)),
    )
    return dataset, shape_names


def evaluate_classifier(
    model: PeakValleyClassifier1D,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate both CNN heads and their joint canonical-shape prediction."""

    model.eval()
    n = peak_ok = valley_ok = joint_ok = 0
    loss_sum = 0.0
    for x, peak_target, valley_target in loader:
        x = x.to(device)
        peak_target = peak_target.to(device)
        valley_target = valley_target.to(device)
        peak_logits, valley_logits = model(x)
        batch_size = x.shape[0]
        loss = F.cross_entropy(peak_logits, peak_target) + F.cross_entropy(
            valley_logits, valley_target
        )
        peak_pred = peak_logits.argmax(dim=1)
        valley_pred = valley_logits.argmax(dim=1)
        peak_matches = peak_pred == peak_target
        valley_matches = valley_pred == valley_target
        n += batch_size
        loss_sum += float(loss) * batch_size
        peak_ok += int(peak_matches.sum())
        valley_ok += int(valley_matches.sum())
        joint_ok += int((peak_matches & valley_matches).sum())
    return {
        "loss": loss_sum / n,
        "peak_accuracy": peak_ok / n,
        "valley_accuracy": valley_ok / n,
        "joint_shape_accuracy": joint_ok / n,
        "n_segments": n,
    }


def train_or_load_classifier(
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    num_workers: int,
    seed: int,
    retrain: bool,
) -> tuple[PeakValleyClassifier1D, dict[str, Any]]:
    """Train the caption-supervised segment CNN, or reuse its derived checkpoint."""

    checkpoint_path = output_dir / "segment_cnn.pth"
    report_path = output_dir / "segment_cnn_report.json"
    model = PeakValleyClassifier1D(segment_len=43).to(device)
    if checkpoint_path.is_file() and report_path.is_file() and not retrain:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
        )
        report = json.loads(report_path.read_text())
        LOGGER.info("Reused segment CNN: %s", checkpoint_path)
        return model.eval(), report

    train_dataset, train_names = _load_split_dataset(data_root, "train")
    valid_dataset, valid_names = _load_split_dataset(data_root, "valid")
    test_dataset, test_names = _load_split_dataset(data_root, "test")
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_args
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.01
    )
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for x, peak_target, valley_target in train_loader:
            x = x.to(device)
            peak_target = peak_target.to(device)
            valley_target = valley_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            peak_logits, valley_logits = model(x)
            loss = F.cross_entropy(peak_logits, peak_target) + F.cross_entropy(
                valley_logits, valley_target
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * x.shape[0]
            n_seen += x.shape[0]
        scheduler.step()
        valid_metrics = evaluate_classifier(model, valid_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / n_seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "valid": valid_metrics,
        }
        history.append(epoch_record)
        accuracy = float(valid_metrics["joint_shape_accuracy"])
        LOGGER.info(
            "CNN epoch %d/%d: train_loss=%.4f, val_joint=%.4f",
            epoch,
            epochs,
            epoch_record["train_loss"],
            accuracy,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("CNN early stopping after epoch %d", epoch)
                break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
    )
    test_metrics = evaluate_classifier(model, test_loader, device)
    report = {
        "model": "PeakValleyClassifier1D",
        "supervision": "segment shape labels parsed from synth-u captions",
        "segment_slices": [[0, 43], [43, 86], [85, 128]],
        "shape_to_heads": {key: list(value) for key, value in SHAPE_TO_TARGET.items()},
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "patience": patience,
            "seed": seed,
        },
        "class_counts": {
            "train": _class_counts(train_names),
            "valid": _class_counts(valid_names),
            "test": _class_counts(test_names),
        },
        "test": test_metrics,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info(
        "CNN test joint shape accuracy: %.4f",
        test_metrics["joint_shape_accuracy"],
    )
    return model.eval(), report


def _class_counts(names: np.ndarray) -> dict[str, int]:
    return {shape: int(np.sum(names == shape)) for shape in SHAPE_NAMES}


@torch.no_grad()
def choose_case_indices(shape_names: np.ndarray, n_cases: int, seed: int) -> list[int]:
    """Choose deterministic, shape-rich test cases with distinct GT tuples."""

    rng = random.Random(seed)
    candidates = list(range(len(shape_names)))
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda index: (
            -sum(shape != "nothing" for shape in shape_names[index]),
            -len(set(shape_names[index])),
        )
    )
    selected: list[int] = []
    seen_patterns: set[tuple[str, ...]] = set()
    for index in candidates:
        pattern = tuple(str(shape) for shape in shape_names[index])
        if pattern in seen_patterns:
            continue
        selected.append(index)
        seen_patterns.add(pattern)
        if len(selected) == n_cases:
            break
    if len(selected) < n_cases:
        raise ValueError(f"requested {n_cases} cases, but only selected {len(selected)}")
    return sorted(selected)


def choose_candidate_indices(shape_names: np.ndarray, pool_size: int, seed: int) -> list[int]:
    """Choose a deterministic shape-rich pool without requiring unique GT tuples."""

    if pool_size < 1 or pool_size > len(shape_names):
        raise ValueError(f"candidate pool size must be within [1, {len(shape_names)}]")
    rng = random.Random(seed)
    candidates = list(range(len(shape_names)))
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda index: (
            -sum(shape != "nothing" for shape in shape_names[index]),
            -len(set(shape_names[index])),
        )
    )
    return candidates[:pool_size]


def score_candidate_cases(
    shape_names: np.ndarray,
    generated: dict[str, np.ndarray],
    predictions: dict[str, list[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Quantify SAE-induced curve changes and CNN target matches per candidate."""

    if len(shape_names) == 0:
        return []
    pure_name = VARIANTS[0][0]
    pure_curves = generated[pure_name]
    scores: list[dict[str, Any]] = []
    for case_index in range(len(shape_names)):
        targets = [str(value) for value in shape_names[case_index]]
        match_counts = {
            name: sum(
                prediction["shape"] == target
                for prediction, target in zip(predictions[name][case_index], targets)
            )
            for name, _, _ in VARIANTS
        }
        pure_scale = max(float(np.std(pure_curves[case_index])), 1e-8)
        normalized_rmse = {
            name: float(
                np.sqrt(np.mean((generated[name][case_index] - pure_curves[case_index]) ** 2))
                / pure_scale
            )
            for name, _, _ in VARIANTS[1:]
        }
        match_delta = {
            name: match_counts[name] - match_counts[pure_name] for name, _, _ in VARIANTS[1:]
        }
        scores.append(
            {
                "match_counts": match_counts,
                "match_delta": match_delta,
                "normalized_rmse": normalized_rmse,
                "max_normalized_rmse": max(normalized_rmse.values()),
                "outcome": (
                    "correct"
                    if max(match_delta.values()) > 0 and min(match_delta.values()) >= 0
                    else "wrong"
                    if min(match_delta.values()) < 0 and max(match_delta.values()) <= 0
                    else "mixed"
                    if min(match_delta.values()) < 0 < max(match_delta.values())
                    else "changed-only"
                ),
            }
        )
    return scores


def select_steering_cases(
    scores: Sequence[dict[str, Any]], n_cases: int
) -> list[int]:
    """Select visually strong correct/wrong cases, balancing both outcomes."""

    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    if n_cases > len(scores):
        raise ValueError(f"requested {n_cases} cases from only {len(scores)} candidates")
    ranked = sorted(
        range(len(scores)),
        key=lambda index: scores[index]["max_normalized_rmse"],
        reverse=True,
    )
    selected: list[int] = []
    per_outcome = max(1, n_cases // 2)
    for outcome in ("correct", "wrong"):
        selected.extend(
            index
            for index in ranked
            if scores[index]["outcome"] == outcome
        )
        selected = list(dict.fromkeys(selected))
        if len([index for index in selected if scores[index]["outcome"] == outcome]) > per_outcome:
            selected = [
                index
                for index in selected
                if scores[index]["outcome"] != outcome
            ] + [
                index
                for index in selected
                if scores[index]["outcome"] == outcome
            ][:per_outcome]
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) == n_cases:
            break
    return selected[:n_cases]


@torch.no_grad()
def generate_variants(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    repo_root: Path,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate all variants with identical per-variant DDIM random streams."""

    condition = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    generated: dict[str, np.ndarray] = {}
    for name, relative_sae_path, _ in VARIANTS:
        if relative_sae_path is None:
            detach_sae(model)
        else:
            attach_sae(model, repo_root / relative_sae_path, device=device)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        samples = model.generate(condition, n_samples=1, sampler="ddim")[0, :, :, 0]
        generated[name] = samples.detach().cpu().float().numpy()
        LOGGER.info("Generated %s in %.1f seconds", name, time.perf_counter() - started)
    detach_sae(model)
    return generated


def render_figures(
    output_dir: Path,
    indices: Sequence[int],
    shape_names: np.ndarray,
    generated: dict[str, np.ndarray],
    predictions: dict[str, list[list[dict[str, Any]]]],
    dpi: int,
    scores: Sequence[dict[str, Any]] | None = None,
) -> list[Path]:
    """Render selected correct/wrong SAE reconstruction cases in one figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if scores is None:
        scores = score_candidate_cases(shape_names, generated, predictions)

    n_columns = 2 if len(indices) > 1 else 1
    n_rows = (len(indices) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18, 5.3 * n_rows + 1.5),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.06, right=0.985, hspace=0.72, wspace=0.16)
    fig.suptitle(
        "Selected SAE reconstruction interventions: correct and wrong CNN-category changes",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.965,
        "Correct/wrong means that the SAE variant improves/degrades the 3-segment "
        "target-match count relative to Pure VerbalTS.",
        ha="center",
        va="top",
        fontsize=10.5,
    )

    pure_name = VARIANTS[0][0]
    for local_index, (dataset_index, axis) in enumerate(zip(indices, axes.flat)):
        gt_labels = [str(value) for value in shape_names[local_index]]
        all_curves = [generated[name][local_index] for name, _, _ in VARIANTS]
        y_min = min(float(np.nanmin(curve)) for curve in all_curves)
        y_max = max(float(np.nanmax(curve)) for curve in all_curves)
        margin = max((y_max - y_min) * 0.08, 0.05)
        score = scores[local_index]

        gt_line = " | ".join(
            f"{stage[0]}={shape}" for stage, shape in zip(STAGE_NAMES, gt_labels)
        )
        annotation_lines = [f"GT: {gt_line}"]
        for name, _, color in VARIANTS:
            axis.plot(
                generated[name][local_index],
                color=color,
                linewidth=1.8,
                label=name,
            )
            cnn_labels = [
                prediction["shape"] for prediction in predictions[name][local_index]
            ]
            cnn_line = " | ".join(
                f"{stage[0]}={shape}" for stage, shape in zip(STAGE_NAMES, cnn_labels)
            )
            if name == pure_name:
                metric = f"match {score['match_counts'][name]}/3"
            else:
                delta = score["match_delta"][name]
                metric = (
                    f"match {score['match_counts'][name]}/3, Δ={delta:+d}, "
                    f"nRMSE={score['normalized_rmse'][name]:.1%}"
                )
            annotation_lines.append(f"{name}: {cnn_line}  [{metric}]")

        outcome = str(score["outcome"]).upper()
        outcome_color = {
            "CORRECT": "#167A3A",
            "WRONG": "#B42318",
            "MIXED": "#7C3AED",
            "CHANGED-ONLY": "#6B7280",
        }[outcome]
        axis.set_title(
            f"Case {dataset_index}\n" + "\n".join(annotation_lines),
            loc="left",
            fontsize=8.5,
            color="#222222",
            pad=9,
            linespacing=1.25,
        )
        axis.text(
            1.0,
            1.31,
            outcome,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=outcome_color,
        )

        axis.axvspan(0, 42, color="#E8F1FA", alpha=0.45, zorder=0)
        axis.axvspan(43, 85, color="#F5ECD8", alpha=0.45, zorder=0)
        axis.axvspan(86, 127, color="#E5F3E7", alpha=0.45, zorder=0)
        axis.axvline(42.5, color="#888888", linewidth=0.8, linestyle=":")
        axis.axvline(85.5, color="#888888", linewidth=0.8, linestyle=":")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Value")
        axis.set_xlim(0, 127)
        axis.set_ylim(y_min - margin, y_max + margin)
        axis.grid(axis="y", alpha=0.2)
        for center, stage in zip((21, 64, 106), STAGE_NAMES):
            axis.text(
                center,
                0.03,
                stage,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#666666",
            )

    for axis in list(axes.flat)[len(indices) :]:
        axis.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)

    png_path = output_dir / "sae_reconstruction_cases.png"
    pdf_path = output_dir / "sae_variant_comparison.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    for old_path in output_dir.glob("case_*.png"):
        old_path.unlink()
    LOGGER.info("Saved combined PNG: %s", png_path)
    LOGGER.info("Saved single-page PDF: %s", pdf_path)
    return [png_path]


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("--indices must contain at least one integer")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/synth-u"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sae_variant_visualization")
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated exact test indices; disables automatic candidate screening",
    )
    parser.add_argument("--n-cases", type=int, default=6)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=128,
        help="Number of shape-rich test cases generated before automatic screening",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--classifier-batch-size", type=int, default=512)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--retrain-classifier", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    LOGGER.info("Using device: %s", device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    classifier, classifier_report = train_or_load_classifier(
        data_root=data_root,
        output_dir=output_dir,
        device=device,
        epochs=args.classifier_epochs,
        batch_size=args.classifier_batch_size,
        learning_rate=args.classifier_lr,
        patience=args.classifier_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        retrain=args.retrain_classifier,
    )

    captions_all = np.load(data_root / "test_text_caps.npy").reshape(-1)
    shape_names_all, _, _ = captions_to_targets(captions_all)
    requested_indices = _parse_indices(args.indices)
    if requested_indices is None:
        candidate_indices = choose_candidate_indices(
            shape_names_all, args.candidate_pool_size, args.seed
        )
    else:
        candidate_indices = requested_indices
    if min(candidate_indices) < 0 or max(candidate_indices) >= len(captions_all):
        raise IndexError(
            f"test indices must be within [0, {len(captions_all) - 1}]: {candidate_indices}"
        )
    LOGGER.info("Generating %d synth-u candidate cases", len(candidate_indices))

    candidate_shape_names = shape_names_all[candidate_indices]
    embedding_path = data_root / "test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy"
    embeddings = np.asarray(
        np.load(embedding_path, mmap_mode="r")[candidate_indices], dtype=np.float32
    )

    verbalts, _ = load_verbalts(
        repo_root / "artifacts/no_sae_pure_verbalts/verbalts.ckpt", device
    )
    generated = generate_variants(verbalts, embeddings, repo_root, device, args.seed)
    candidate_predictions = {
        name: classify_curves(classifier, generated[name], device) for name, _, _ in VARIANTS
    }
    candidate_scores = score_candidate_cases(
        candidate_shape_names, generated, candidate_predictions
    )
    if requested_indices is None:
        selected_local_indices = select_steering_cases(candidate_scores, args.n_cases)
    else:
        selected_local_indices = list(range(len(candidate_indices)))

    indices = [candidate_indices[index] for index in selected_local_indices]
    generated = {
        name: curves[selected_local_indices] for name, curves in generated.items()
    }
    predictions = {
        name: [values[index] for index in selected_local_indices]
        for name, values in candidate_predictions.items()
    }
    scores = [candidate_scores[index] for index in selected_local_indices]
    captions = captions_all[indices]
    shape_names = shape_names_all[indices]
    ground_truth = np.asarray(
        np.load(data_root / "test_ts.npy", mmap_mode="r")[indices, :, 0], dtype=np.float32
    )
    LOGGER.info("Selected synth-u cases: %s", indices)

    png_paths = render_figures(
        output_dir=output_dir,
        indices=indices,
        shape_names=shape_names,
        generated=generated,
        predictions=predictions,
        dpi=args.dpi,
        scores=scores,
    )

    np.savez_compressed(
        output_dir / "curves.npz",
        indices=np.asarray(indices),
        ground_truth=ground_truth,
        pure_verbalts=generated["Pure VerbalTS"],
        sae_t5_45=generated["Top-K SAE (t=5–45)"],
    )
    summary = {
        "seed": args.seed,
        "sampler": "ddim",
        "shared_random_stream_across_variants": True,
        "intervention": (
            "SAE reconstruction replacement at residual layer 1; no latent feature "
            "direction or target-category steering is applied"
        ),
        "selection": {
            "automatic": requested_indices is None,
            "candidate_pool_size": len(candidate_indices),
            "candidate_indices": candidate_indices,
            "n_selected": len(indices),
            "criterion": (
                "balance correct/wrong CNN target-match deltas, then rank by maximum "
                "normalized RMSE versus Pure VerbalTS"
            ),
        },
        "segment_slices": [[0, 43], [43, 86], [85, 128]],
        "classifier_test": classifier_report.get("test", {}),
        "cases": [],
    }
    for local_index, dataset_index in enumerate(indices):
        summary["cases"].append(
            {
                "test_index": dataset_index,
                "caption": str(captions[local_index]),
                "fine_grained_gt": {
                    stage: str(shape)
                    for stage, shape in zip(STAGE_NAMES, shape_names[local_index])
                },
                "cnn_generated_predictions": {
                    name: {
                        stage: prediction
                        for stage, prediction in zip(
                            STAGE_NAMES, predictions[name][local_index]
                        )
                    }
                    for name, _, _ in VARIANTS
                },
                "screening_score": scores[local_index],
                "figure": str(png_paths[0]),
            }
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Saved summary: %s", summary_path)
    for path in png_paths:
        LOGGER.info("Saved figure: %s", path)


if __name__ == "__main__":
    main()
