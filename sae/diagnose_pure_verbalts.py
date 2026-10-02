"""Stage-A diagnostics for a trained pure VerbalTS checkpoint.

All paired comparisons restore the same RNG state before sampling.  The
fixed-noise swap uses captions already present in the validation set: the
global sentences and beginning/end morphology sentences must match exactly,
while the middle morphology spans all four classes.  This avoids silently
mixing embeddings produced by a different text encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import compute_cnn_report, generate_variant
from sae.provenance import runtime_metadata, sha256_file, write_json
from sae.shapes import SHAPE_NAMES, captions_to_targets, classify_curves, parse_segment_shapes


def curve_statistics(curves: np.ndarray) -> dict[str, Any]:
    """Return distribution and per-curve amplitude summaries."""
    x = np.asarray(curves, dtype=np.float64)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"expected (N,L[,1]), got {x.shape}")
    ranges = np.ptp(x, axis=1)
    extrema = np.max(np.abs(x), axis=1)
    return {
        "mean": float(x.mean()), "variance": float(x.var()),
        "minimum": float(x.min()), "maximum": float(x.max()),
        "range_quantiles": quantiles(ranges),
        "absolute_extreme_quantiles": quantiles(extrema),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)}


def mse_summary(curves: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    per = np.mean((np.asarray(curves) - np.asarray(truth)) ** 2, axis=1)
    return {"mean": float(per.mean()), "quantiles": quantiles(per),
            "worst_rows": np.argsort(per)[-10:][::-1].astype(int).tolist()}


def segment_correct(names: np.ndarray, predictions: list[list[dict[str, Any]]]) -> np.ndarray:
    return np.asarray([[predictions[i][j]["shape"] == names[i, j] for j in range(3)]
                       for i in range(len(names))], dtype=np.float64)


def paired_accuracy_difference(
    correct: np.ndarray, shuffled: np.ndarray, seed: int, n_boot: int = 10000
) -> dict[str, Any]:
    """Paired bootstrap CI for per-curve mean segment accuracy difference."""
    deltas = correct.mean(axis=1) - shuffled.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[draws].mean(axis=1)
    return {"mean": float(deltas.mean()),
            "bootstrap_ci95": np.quantile(means, (.025, .975)).astype(float).tolist(),
            "replicates": n_boot}


def condition_comparison(
    target_names: np.ndarray,
    input_names: np.ndarray,
    correct_predictions: list[list[dict[str, Any]]],
    shuffled_predictions: list[list[dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    """Score degradation against original targets, separately from input adherence."""
    correct = segment_correct(target_names, correct_predictions)
    shuffled_vs_original = segment_correct(target_names, shuffled_predictions)
    shuffled_vs_input = segment_correct(input_names, shuffled_predictions)
    segment = paired_accuracy_difference(correct, shuffled_vs_original, seed)
    exact = paired_accuracy_difference(
        correct.all(axis=1, keepdims=True),
        shuffled_vs_original.all(axis=1, keepdims=True), seed + 1,
    )
    return {
        # Preserve the original keys consumed by the admission checker.  Their
        # semantics are now explicitly correct-vs-shuffled on original labels.
        **segment,
        "scoring_target": "original_condition",
        "segment_accuracy": segment,
        "whole_curve_exact_match": exact,
        "shuffled_original_target_segment_accuracy": float(shuffled_vs_original.mean()),
        "shuffled_original_target_exact_match": float(
            shuffled_vs_original.all(axis=1).mean()
        ),
        "shuffled_input_adherence_segment_accuracy": float(shuffled_vs_input.mean()),
        "shuffled_input_adherence_exact_match": float(shuffled_vs_input.all(axis=1).mean()),
    }


def caption_swap_groups(captions: np.ndarray) -> list[dict[str, int]]:
    """Find exact-context caption quartets indexed by middle shape."""
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for index, raw in enumerate(np.asarray(captions).reshape(-1)):
        caption = str(raw)
        shapes = parse_segment_shapes(caption)
        sentences = [s.strip() for s in caption.split(".") if s.strip()]
        context = tuple(s for s in sentences if "middle" not in s.lower())
        key = (". ".join(context), shapes[0], shapes[2])
        groups.setdefault(key, {}).setdefault(shapes[1], index)
    wanted = set(SHAPE_NAMES)
    return [group for group in groups.values() if set(group) == wanted]


def _generate_batches(
    model, embeddings: np.ndarray, seed: int, batch_size: int,
    n_samples: int = 1,
    sampler: str = "ddim", guidance_scale: float | None = None,
    trace: list[dict[str, float]] | None = None,
) -> np.ndarray:
    outputs = []
    device = next(model.parameters()).device
    for start in range(0, len(embeddings), batch_size):
        condition = torch.from_numpy(embeddings[start:start + batch_size]).to(device)
        torch.manual_seed(seed + start)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + start)
        kwargs = {}
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if trace is not None:
            def record(step, x, pred_noise, x0_pred):
                row = {
                    "batch_start": start, "timestep": step,
                    "x_variance": float(x.float().var(unbiased=False).cpu()),
                    "x_absolute_max": float(x.float().abs().max().cpu()),
                    "pred_noise_variance": float(pred_noise.float().var(unbiased=False).cpu()),
                    "pred_noise_absolute_max": float(pred_noise.float().abs().max().cpu()),
                }
                if x0_pred is not None:
                    row.update({
                        "x0_variance": float(x0_pred.float().var(unbiased=False).cpu()),
                        "x0_absolute_max": float(x0_pred.float().abs().max().cpu()),
                    })
                trace.append(row)
            kwargs["trace_callback"] = record
        generated = model.generate(
            condition, n_samples=n_samples, sampler=sampler, **kwargs
        )
        outputs.append(generated[:, :, :, 0].cpu().float().numpy())
    return np.concatenate(outputs, axis=1)  # (draw, sample, length)


def _generate_shared_noise(
    model, embeddings: np.ndarray, seed: int,
    sampler: str = "ddim", guidance_scale: float | None = None,
) -> np.ndarray:
    """Generate each condition after restoring exactly the same RNG seed."""
    device = next(model.parameters()).device
    outputs = []
    for embedding in embeddings:
        outputs.append(_generate_batches(
            model, embedding[None], seed, 1, 1,
            sampler=sampler, guidance_scale=guidance_scale,
        )[0, 0])
    return np.stack(outputs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--classifier-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    p.add_argument(
        "--guidance-scale", type=float, default=None,
        help="Override checkpoint CFG scale (use 1 for conditional prediction without CFG)",
    )
    p.add_argument("--trace-reverse", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = args.data_root
    attrs_all = np.load(root / "valid_attrs_idx.npy")
    captions_all = np.load(root / "valid_text_caps.npy", allow_pickle=True).reshape(-1)
    embeddings_all = np.load(root / "valid_cap_emb.npy").astype(np.float32)
    truth_all = np.load(root / "valid_ts.npy")[:, :, 0]
    if not (len(attrs_all) == len(captions_all) == len(embeddings_all) == len(truth_all)):
        raise ValueError("validation arrays are not aligned")
    attr_names = np.asarray(SHAPE_NAMES)[attrs_all]
    parsed, _, _ = captions_to_targets(captions_all)
    captions_encode_morphology = np.array_equal(parsed, attr_names)
    names_all = attr_names
    peak_map, valley_map = np.asarray([0, 1, 2, 0]), np.asarray([0, 0, 0, 1])
    peaks_all, valleys_all = peak_map[attrs_all], valley_map[attrs_all]

    indices = np.random.default_rng(2026).permutation(len(attrs_all))[:args.n_samples]
    embeddings, truth, attrs = embeddings_all[indices], truth_all[indices], attrs_all[indices]
    names, peaks, valleys = names_all[indices], peaks_all[indices], valleys_all[indices]
    shuffled_order = np.random.default_rng(args.seed + 1).permutation(len(indices))
    device = torch.device(args.device)
    model, model_hash = load_verbalts(args.model_checkpoint, device)
    if not captions_encode_morphology:
        raise ValueError("captions must encode morphology aligned with attrs_idx")
    cnn = PeakValleyClassifier1D(segment_len=43).to(device)
    cnn.load_state_dict(torch.load(args.classifier_checkpoint, map_location=device, weights_only=True))
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)

    trace: list[dict[str, float]] | None = [] if args.trace_reverse else None
    generation_kwargs = {"sampler": args.sampler, "guidance_scale": args.guidance_scale}
    correct = _generate_batches(
        model, embeddings, args.seed, args.batch_size, 1,
        trace=trace, **generation_kwargs,
    )[0]
    # Restoring identical per-batch RNG seeds guarantees paired initial noise
    # (and paired DDPM reverse noise) for the caption permutation.
    text_shuffled = _generate_batches(
        model, embeddings[shuffled_order], args.seed, args.batch_size, 1,
        **generation_kwargs,
    )[0]
    ten = _generate_batches(
        model, embeddings, args.seed, args.batch_size, 10, **generation_kwargs
    )
    np.save(args.output_dir / "correct.npy", correct)
    np.save(args.output_dir / "text_shuffled.npy", text_shuffled)
    best_per = ((ten - truth[None]) ** 2).mean(axis=2).min(axis=0)

    def cnn_report(curves: np.ndarray, target_names=names, target_peaks=peaks,
                   target_valleys=valleys) -> dict[str, Any]:
        return compute_cnn_report(target_names, target_peaks, target_valleys,
                                  classify_curves(cnn, curves, device))

    zeros = np.zeros_like(truth)
    train_mean = float(np.load(root / "train_ts.npy", mmap_mode="r").mean())
    correct_predictions = classify_curves(cnn, correct, device)
    variant_curves = {"text_shuffled": text_shuffled}
    variant_predictions = {
        key: classify_curves(cnn, curves, device) for key, curves in variant_curves.items()
    }
    shuffled_predictions = variant_predictions["text_shuffled"]
    correct_cnn = compute_cnn_report(names, peaks, valleys, correct_predictions)
    input_names_by_variant = {"text_shuffled": names[shuffled_order]}
    comparisons = {
        key: condition_comparison(
            names, input_names_by_variant[key], correct_predictions, predictions,
            args.seed + 2 + offset * 2,
        )
        for offset, (key, predictions) in enumerate(variant_predictions.items())
    }
    shuffled_cnn_original = compute_cnn_report(
        names, peaks, valleys, shuffled_predictions
    )
    shuffled_cnn_input = compute_cnn_report(
        names[shuffled_order], peaks[shuffled_order], valleys[shuffled_order],
        shuffled_predictions,
    )
    report: dict[str, Any] = {
        "indices": indices.tolist(), "seed": args.seed,
        "generation": {
            "sampler": args.sampler,
            "guidance_scale": model.cfg_scale if args.guidance_scale is None
                              else args.guidance_scale,
            "num_steps": model.num_steps,
            "betas": {"first": float(model.betas[0].cpu()),
                      "last": float(model.betas[-1].cpu())},
        },
        "model_sha256": model_hash,
        "files": {p.name: sha256_file(p) for p in (
            root / "valid_attrs_idx.npy", root / "valid_text_caps.npy",
            root / "valid_cap_emb.npy", root / "valid_ts.npy")},
        "runtime": runtime_metadata(Path(__file__).resolve().parents[1]),
        "classifier_sha256": sha256_file(args.classifier_checkpoint),
        "baselines": {
            "zero": mse_summary(zeros, truth),
            "train_scalar_mean": mse_summary(np.full_like(truth, train_mean), truth),
        },
        "correct": {"mse": mse_summary(correct, truth), "cnn": correct_cnn},
        "shuffled": {
            "mse_vs_original_truth": mse_summary(text_shuffled, truth),
            "cnn_vs_original_condition": shuffled_cnn_original,
            "cnn_vs_input_condition": shuffled_cnn_input,
            "mean_absolute_output_change": float(np.abs(correct - text_shuffled).mean()),
        },
        "condition_sensitivity": comparisons["text_shuffled"],
        "condition_sensitivity_by_shuffle": comparisons,
        "best_of_10": {"mse_mean": float(best_per.mean()), "mse_quantiles": quantiles(best_per)},
        "distribution": {"truth": curve_statistics(truth), "generated": curve_statistics(correct)},
    }
    if trace is not None:
        report["reverse_trace"] = trace

    groups = caption_swap_groups(captions_all) if captions_encode_morphology else []
    swap_rows = []
    for group_id, group in enumerate(groups[:min(16, len(groups))]):
        order = [group[name] for name in SHAPE_NAMES]
        curves = _generate_shared_noise(
            model, embeddings_all[order], args.seed + 10000 + group_id,
            **generation_kwargs,
        )
        target_names = names_all[order]
        target_peaks, target_valleys = peaks_all[order], valleys_all[order]
        predictions = classify_curves(cnn, curves, device)
        swap_rows.append({"source_indices": order, "target_middle": list(SHAPE_NAMES),
                          "cnn": compute_cnn_report(target_names, target_peaks, target_valleys, predictions),
                          "middle_predictions": [row[1]["shape"] for row in predictions],
                          "pairwise_curve_mse": float(np.mean((curves[:, None] - curves[None, :]) ** 2))})
    report["fixed_noise_caption_swap"] = {"eligible_groups": len(groups), "evaluated": len(swap_rows),
                                           "groups": swap_rows}
    write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()