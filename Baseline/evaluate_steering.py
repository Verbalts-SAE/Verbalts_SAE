"""Evaluate the DiffLens-style multiplicative IG-steering baseline.

Paired variants share one DDIM random stream:

* ``pure``               — Pure VerbalTS, no SAE wrapper;
* ``difflens_steered``   — SAE inside its timestep window with the fixed
  multiplicative IG edit (boost the top-k positive-IG dims, suppress the
  top-k negative-IG dims) applied to samples whose caption targets the
  located class (default: Beginning / single peak).

Reported metrics: MSE (overall, paired delta vs pure with bootstrap 95% CI),
the frozen segment CNN's segment accuracy, per-stage accuracy and
whole-curve exact match, plus a target-class-focused breakdown (stage/shape
accuracy and MSE over the beginning segment of target-class samples).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

# Make the repository root importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import contsg.models.verbalts  # noqa: F401, E402  (Registry side effect)
from contsg.eval.metrics.segment import PeakValleyClassifier1D  # noqa: E402
from sae.activations import load_verbalts  # noqa: E402
from sae.eval_guidance import (  # noqa: E402
    compute_cnn_report,
    compute_mse_report,
    generate_all_variants,
)
from sae.provenance import (  # noqa: E402
    runtime_metadata,
    sha256_file,
    tensor_state_sha256,
    write_json,
)
from sae.shapes import (  # noqa: E402
    SHAPE_NAMES,
    STAGE_NAMES,
    captions_to_targets,
    classify_curves,
    split_segments,
)
from sae.steering import load_sae_checkpoint  # noqa: E402

from Baseline.ig_attribution import load_located_features  # noqa: E402
from Baseline.multiplicative_steer import MultiplicativeSteeringWrapper  # noqa: E402

# ``generate_all_variants`` special-cases its module-level PURE_KEY, which is
# exactly "pure"; keeping the same key here reuses that path.
PURE_KEY = "pure"
STEERED_KEY = "difflens_steered"


def compute_beginning_report(
    curves: np.ndarray,
    ground_truth: np.ndarray,
    shape_names: np.ndarray,
    target_shape: str,
) -> dict[str, Any]:
    """MSE over the beginning segment (all samples + target-class samples)."""

    mask = shape_names[:, 0] == target_shape
    curve_segments = split_segments(curves)  # (N, 3, 43)
    ground_truth_segments = split_segments(ground_truth)
    beginning = curve_segments[:, 0]
    beginning_ground_truth = ground_truth_segments[:, 0]
    mse_all = float(np.mean((beginning - beginning_ground_truth) ** 2))
    mse_target = (
        float(np.mean((beginning[mask] - beginning_ground_truth[mask]) ** 2))
        if mask.any()
        else float("nan")
    )
    return {
        "beginning_segment_mse": mse_all,
        "beginning_segment_mse_target_class": mse_target,
        "target_class_samples": int(mask.sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=root / "artifacts/no_sae_pure_verbalts/verbalts.ckpt",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        default=root / "results/sae_retrain/models/t5_45/best.pt",
    )
    parser.add_argument(
        "--located-features",
        type=Path,
        default=(
            root
            / "results/baseline_difflens_ig"
            / "located_beginning_single_peak.json"
        ),
        help="JSON artifact written by locate_features.py",
    )
    parser.add_argument(
        "--boost-factor",
        type=float,
        default=2.0,
        help="Multiplier applied to the top-k positive-IG dims",
    )
    parser.add_argument(
        "--suppress-factor",
        type=float,
        default=0.5,
        help="Multiplier applied to the top-k negative-IG dims",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="Upper clamp for merged multiplicative ratios (default: max(1, boost-factor))",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=None,
        help="Lower clamp for merged multiplicative ratios (default: min(1, suppress-factor))",
    )
    parser.add_argument(
        "--evaluation-cnn",
        type=Path,
        default=(
            root
            / "results/sae_retrain/classwise_steering/evaluation_classifier"
            / "segment_cnn.pth"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=root / "datasets/synth-u")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/baseline_difflens_ig/eval"
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Default: full test set")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Reuse saved chunks")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    repo_root = _REPO_ROOT
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Located features ----------------------------------------------------
    located_path = args.located_features.resolve()
    located = load_located_features(located_path)
    target = located["target"]
    stage = int(target["stage"])
    shape = int(target["shape"])
    shape_name = str(target["shape_name"])
    group_key = str(target["group"])
    topk = int(located["topk"])
    edits = {
        group_key: {
            "boost": [int(dim) for dim in located["boost_dims"]],
            "suppress": [int(dim) for dim in located["suppress_dims"]],
        }
    }
    if len(edits[group_key]["boost"]) != topk or len(edits[group_key]["suppress"]) != topk:
        raise ValueError(
            f"located features expect {topk} boost and {topk} suppress dims"
        )
    if shape_name not in SHAPE_NAMES:
        raise ValueError(f"unknown located shape name {shape_name!r}")

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data and ground-truth tuples ----------------------------------------
    captions = np.load(data_root / "test_text_caps.npy", allow_pickle=True).reshape(-1)
    shape_names_all, peak_all, valley_all = captions_to_targets(captions)
    total = len(captions)
    start = args.sample_offset
    limit = total if args.n_samples is None else min(args.n_samples, total - start)
    if start < 0 or limit < 1 or start + limit > total:
        raise ValueError(
            f"sample range [{start}, {start + limit}) is invalid for test size {total}"
        )
    indices = np.arange(start, start + limit)
    captions = captions[indices]
    shape_names = shape_names_all[indices]
    peak_targets = peak_all[indices]
    valley_targets = valley_all[indices]
    print(f"evaluating {limit} samples")

    embeddings = np.asarray(
        np.load(
            data_root / "test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy",
            mmap_mode="r",
        )[indices],
        dtype=np.float32,
    )
    ground_truth = np.asarray(
        np.load(data_root / "test_ts.npy", mmap_mode="r")[indices, :, 0],
        dtype=np.float32,
    )

    # --- Frozen evaluation CNN ------------------------------------------------
    classifier_path = args.evaluation_cnn.resolve()
    classifier = PeakValleyClassifier1D(segment_len=43).to(device)
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True), strict=True
    )
    classifier.eval()
    classifier_report_path = classifier_path.with_name("segment_cnn_report.json")
    classifier_report = (
        json.loads(classifier_report_path.read_text(encoding="utf-8"))
        if classifier_report_path.is_file()
        else {}
    )

    # --- VerbalTS + multiplicative steering wrapper ---------------------------
    model_path = args.model_checkpoint.resolve()
    model, model_state_hash = load_verbalts(model_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sae_path = args.sae_checkpoint.resolve()
    sae, t_range, sae_checkpoint = load_sae_checkpoint(sae_path, device)
    wrapper = MultiplicativeSteeringWrapper(
        sae,
        t_range,
        edits,
        boost_factor=args.boost_factor,
        suppress_factor=args.suppress_factor,
        max_ratio=args.max_ratio,
        min_ratio=args.min_ratio,
    ).to(device).eval()
    wrappers = {STEERED_KEY: wrapper}
    variant_keys = (PURE_KEY, STEERED_KEY)
    labels = {
        PURE_KEY: "Pure VerbalTS",
        STEERED_KEY: (
            f"DiffLens IG multiplicative steering ({group_key}, "
            f"boost x{args.boost_factor:g}, suppress x{args.suppress_factor:g})"
        ),
    }

    # --- Generation ------------------------------------------------------------
    wrapper.reset_audit()
    generation_args = SimpleNamespace(
        resume=args.resume, batch_size=args.batch_size, seed=args.seed
    )
    curves = generate_all_variants(
        model,
        embeddings,
        shape_names,
        wrappers,
        generation_args,
        output_dir,
        device,
        variant_keys,
    )
    for key in variant_keys:
        if curves[key].shape != ground_truth.shape:
            raise ValueError(
                f"generated {key} shape {curves[key].shape} != "
                f"ground-truth {ground_truth.shape}"
            )
    predictions = {
        key: classify_curves(classifier, curves[key], device) for key in variant_keys
    }

    # --- Metrics -----------------------------------------------------------------
    mse_reports = {
        key: compute_mse_report(curves[key], ground_truth, curves[PURE_KEY], args.seed)
        for key in variant_keys
    }
    cnn_reports = {
        key: compute_cnn_report(
            shape_names, peak_targets, valley_targets, predictions[key]
        )
        for key in variant_keys
    }
    beginning_reports = {
        key: compute_beginning_report(
            curves[key], ground_truth, shape_names, shape_name
        )
        for key in variant_keys
    }
    stage_shape_key = f"{STAGE_NAMES[stage]}/{shape_name}"

    # --- Artifacts ----------------------------------------------------------------
    save_kwargs: dict[str, Any] = {
        "indices": np.asarray(indices),
        "ground_truth": ground_truth,
    }
    for key in variant_keys:
        save_kwargs[key] = curves[key]
    np.savez_compressed(output_dir / "curves.npz", **save_kwargs)

    per_case = []
    for local_index, dataset_index in enumerate(indices):
        per_case.append(
            {
                "test_index": int(dataset_index),
                "caption": str(captions[local_index]),
                "target_tuple": [str(shape) for shape in shape_names[local_index]],
                "cnn_predictions": {
                    key: predictions[key][local_index] for key in variant_keys
                },
            }
        )
    write_json(output_dir / "per_case.json", per_case)

    steering_formula = (
        f"z_new[d] = z[d] * {args.boost_factor:g} for the top-{topk} positive-IG "
        f"dims, z[d] * {args.suppress_factor:g} for the top-{topk} negative-IG "
        f"dims of {group_key!r}; applied at every in-window diffusion step to "
        "samples whose caption targets that class; other samples keep the SAE "
        "reconstruction"
    )
    summary = {
        "method": "DiffLens-style multiplicative IG steering",
        "steering_formula": steering_formula,
        "located_features": str(located_path),
        "located_features_sha256": sha256_file(located_path),
        "target": dict(located["target"]),
        "topk": topk,
        "boost_factor": args.boost_factor,
        "suppress_factor": args.suppress_factor,
        "t_range": list(t_range),
        "sampler": "ddim",
        "seed": args.seed,
        "shared_random_stream_across_variants": True,
        "sample_range": [int(start), int(start + limit)],
        "n_samples": int(limit),
        "mse": mse_reports,
        "cnn": cnn_reports,
        "beginning_segment": beginning_reports,
        "hook_audits": {STEERED_KEY: wrapper.audit()},
        "evaluation_classifier_test": classifier_report.get("test", {}),
        "audit": {
            "model_checkpoint": str(model_path),
            "model_checkpoint_sha256": sha256_file(model_path),
            "model_state_sha256": model_state_hash,
            "sae_checkpoint": str(sae_path),
            "sae_checkpoint_sha256": sha256_file(sae_path),
            "sae_state_sha256": sae_checkpoint.get(
                "state_dict_sha256", tensor_state_sha256(sae.state_dict())
            ),
            "evaluation_classifier_checkpoint": str(classifier_path),
            "evaluation_classifier_checkpoint_sha256": sha256_file(classifier_path),
            "runtime": runtime_metadata(repo_root),
        },
    }
    write_json(output_dir / "summary.json", summary)

    # --- Comparison tables ------------------------------------------------------
    rows = []
    for key in variant_keys:
        rows.append(
            {
                "variant": labels[key],
                "mse_overall": mse_reports[key]["overall_mse"],
                "mse_paired_delta_vs_pure": mse_reports[key][
                    "paired_delta_vs_pure_mean"
                ],
                "mse_delta_ci95_low": mse_reports[key]["paired_delta_vs_pure_ci95"][0],
                "mse_delta_ci95_high": mse_reports[key]["paired_delta_vs_pure_ci95"][1],
                "segment_accuracy": cnn_reports[key]["segment_accuracy"],
                f"{stage_shape_key}_accuracy": cnn_reports[key][
                    "stage_shape_accuracy"
                ].get(stage_shape_key, float("nan")),
                "whole_curve_exact_match": cnn_reports[key]["whole_curve_exact_match"],
                "beginning_segment_mse": beginning_reports[key][
                    "beginning_segment_mse"
                ],
                "beginning_segment_mse_target_class": beginning_reports[key][
                    "beginning_segment_mse_target_class"
                ],
                "beginning_segment_mse_delta_vs_pure": (
                    beginning_reports[key]["beginning_segment_mse"]
                    - beginning_reports[PURE_KEY]["beginning_segment_mse"]
                ),
            }
        )
    fieldnames = list(rows[0].keys())
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_lines = [
        "| " + " | ".join(fieldnames) + " |",
        "|" + "|".join("---" for _ in fieldnames) + "|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
                for key in fieldnames
            )
            + " |"
        )
    (output_dir / "comparison.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"mse": mse_reports, "cnn": cnn_reports}, indent=2, default=str
        )
    )
    print(f"saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
