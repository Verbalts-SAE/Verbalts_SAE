"""Evaluate per-sample latent classifier guidance on the full synth-u test set.

Paired variants share one DDIM random stream:

* ``pure``        — Pure VerbalTS, no SAE wrapper;
* ``t5_45_g{eta}`` — SAE at diffusion steps 5–45 with the shape-aware hybrid
  latent classifier guidance at strength ``eta``.

The winning intervention is the shape-aware hybrid recipe: confidence-weighted
per-sample gradient ascent on the caption-derived B/M/E target under the
position-aware latent classifier, with full-strength guidance reserved for the
hard shapes (``double peaks``, ``sag``).  See
``results/compositional_guidance_hybrid_final.md`` for the complete gamma×eta dose
matrix and multi-seed robustness results.

Reported metrics are CTTP, MSE (overall, per-sample, paired delta versus
Pure with bootstrap 95% CI) and the frozen segment CNN's segment accuracy,
per-stage accuracy and whole-curve exact match.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.shape_classifier import load_classifier_checkpoint
from sae.provenance import runtime_metadata, sha256_file, tensor_state_sha256, write_json
from sae.shapes import (
    SHAPE_NAMES,
    SHAPE_TO_TARGET,
    STAGE_NAMES,
    captions_to_targets,
    classify_curves,
)
from sae.steering import LatentClassifierGuidanceWrapper, load_sae_checkpoint

# Import for the Registry side effect.
import contsg.models.verbalts  # noqa: F401, E402


WINDOW_LABELS = {
    "t40_45": "SAE t=40–45 classifier guidance",
    "t0_45": "SAE t=0–45 classifier guidance",
    "t5_45": "SAE t=5–45 classifier guidance",
}
PURE_KEY = "pure"
CHUNK_SIZE = 250
BOOTSTRAP_REPLICATES = 10000


def paired_bootstrap_ci(
    deltas: np.ndarray, seed: int, n_boot: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    """Paired bootstrap 95% CI of the mean of ``deltas``."""

    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n < 2:
        return float(np.nan), float(np.nan)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = deltas[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_mse_report(
    curves: np.ndarray, ground_truth: np.ndarray, pure: np.ndarray, seed: int
) -> dict[str, Any]:
    """Overall/per-sample MSE plus paired delta and bootstrap CI versus Pure."""

    per_sample = np.mean((curves - ground_truth) ** 2, axis=1)
    per_sample_pure = np.mean((pure - ground_truth) ** 2, axis=1)
    deltas = per_sample - per_sample_pure
    ci_low, ci_high = paired_bootstrap_ci(deltas, seed)
    return {
        "overall_mse": float(np.mean((curves - ground_truth) ** 2)),
        "per_sample_mse_mean": float(per_sample.mean()),
        "per_sample_mse_std": float(per_sample.std()),
        "paired_delta_vs_pure_mean": float(deltas.mean()),
        "paired_delta_vs_pure_ci95": [ci_low, ci_high],
    }


def compute_cnn_report(
    shape_names: np.ndarray,
    peak_targets: np.ndarray,
    valley_targets: np.ndarray,
    predictions: Sequence[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Segment / per-stage / whole-curve CNN metrics for one variant."""

    n_curves = len(shape_names)
    correct = np.zeros((n_curves, len(STAGE_NAMES)), dtype=bool)
    target_probabilities = np.zeros((n_curves, len(STAGE_NAMES)), dtype=np.float64)
    for curve_index in range(n_curves):
        for stage in range(len(STAGE_NAMES)):
            prediction = predictions[curve_index][stage]
            shape = str(shape_names[curve_index, stage])
            correct[curve_index, stage] = prediction["shape"] == shape
            peak_id, valley_id = SHAPE_TO_TARGET[shape]
            target_probabilities[curve_index, stage] = float(
                prediction["peak_probabilities"][peak_id]
                * prediction["valley_probabilities"][valley_id]
            )
    non_nothing = shape_names != "nothing"
    stage_shape_accuracy: dict[str, float] = {}
    for stage in range(len(STAGE_NAMES)):
        for shape in SHAPE_NAMES:
            selected = shape_names[:, stage] == shape
            stage_shape_accuracy[f"{STAGE_NAMES[stage]}/{shape}"] = (
                float(correct[selected, stage].mean()) if selected.any() else float("nan")
            )
    return {
        "segment_accuracy": float(correct.mean()),
        "per_stage_accuracy": {
            STAGE_NAMES[stage]: float(correct[:, stage].mean())
            for stage in range(len(STAGE_NAMES))
        },
        "whole_curve_exact_match": float(correct.all(axis=1).mean()),
        "non_nothing_segment_accuracy": (
            float(correct[non_nothing].mean()) if non_nothing.any() else float("nan")
        ),
        "per_shape_recall": {
            shape: (
                float(correct[shape_names == shape].mean())
                if (shape_names == shape).any()
                else float("nan")
            )
            for shape in SHAPE_NAMES
        },
        "stage_shape_accuracy": stage_shape_accuracy,
        "mean_target_probability": float(target_probabilities.mean()),
    }


def load_cttp_embedder(
    config_path: Path, checkpoint_path: Path, device: torch.device, output_dir: Path
):
    """Load the released synth-u CTTP model, resolving ``${LONGCLIP_ROOT}``."""

    try:
        from contsg.eval.embedder import CLIPEmbedder
    except Exception as exc:  # noqa: BLE001
        return None, f"CTTP embedder import failed: {type(exc).__name__}: {exc}"

    if not config_path.is_file() or not checkpoint_path.is_file():
        return None, (
            f"CTTP resources missing: {config_path} or {checkpoint_path}; "
            "run sae/download_cttp_resources.py first"
        )
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        config_path.parent / "longclip",
        repo_root / "checkpoints" / "longclip",
    ]
    longclip_root = next(
        (candidate for candidate in candidates if candidate.is_dir()),
        candidates[-1],
    )
    raw = config_path.read_text(encoding="utf-8")
    resolved = raw.replace("${LONGCLIP_ROOT}", str(longclip_root.resolve()))
    local_config = output_dir / "cttp_model_configs.yaml"
    local_config.write_text(resolved, encoding="utf-8")
    if not longclip_root.is_dir():
        return None, f"LongCLIP directory missing: {longclip_root}"
    try:
        return CLIPEmbedder(local_config, checkpoint_path, device), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"CTTP embedder load failed: {type(exc).__name__}: {exc}"


@torch.no_grad()
def compute_cttp(
    embedder,
    curves: np.ndarray,
    captions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    """``mean_i <E_ts(x_i), E_text(c_i)>`` using the released CTTP model."""

    tensors = torch.from_numpy(curves[:, :, None].astype(np.float32))
    text_captions = [str(caption) for caption in captions.reshape(-1)]
    ts_embeddings = []
    for start in range(0, len(curves), batch_size):
        chunk = tensors[start:start + batch_size].to(device)
        ts_embeddings.append(
            embedder.get_ts_embedding(chunk, torch.full((chunk.shape[0],), 128)).cpu()
        )
    ts_emb = torch.cat(ts_embeddings, dim=0)
    cap_embeddings = []
    for start in range(0, len(text_captions), batch_size):
        cap_embeddings.append(
            embedder.get_text_embedding({"cap": text_captions[start:start + batch_size]}).cpu()
        )
    cap_emb = torch.cat(cap_embeddings, dim=0)
    return float((ts_emb * cap_emb).sum(dim=1).mean())


def generate_variant(
    model: torch.nn.Module,
    condition: torch.Tensor,
    wrapper: torch.nn.Module | None,
    seed: int,
    sampler: str = "ddim",
    dynamic_threshold: float | None = None,
    dynamic_threshold_quantile: float = 0.995,
) -> np.ndarray:
    """Generate one curve per condition under an optional SAE wrapper.

    Attaches ``wrapper`` as the VerbalTS residual activation transform (None
    restores the Pure path), seeds the global RNG state, and returns the
    generated (B, 128) curves as a NumPy array.
    """

    model.verbalts.activation_transform = wrapper
    torch.manual_seed(seed)
    if condition.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generated = model.generate(
        condition,
        n_samples=1,
        sampler=sampler,
        dynamic_threshold=dynamic_threshold,
        dynamic_threshold_quantile=dynamic_threshold_quantile,
    )[0, :, :, 0]
    return generated.detach().cpu().float().numpy()


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=root / "artifacts/no_sae_pure_verbalts/verbalts.ckpt",
    )
    parser.add_argument("--results-root", type=Path, default=root / "results/sae_retrain")
    parser.add_argument("--data-root", type=Path, default=root / "datasets/synth-u")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/compositional_full_eval"
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=(
            root
            / "results/sae_retrain/classwise_steering/evaluation_classifier/segment_cnn.pth"
        ),
    )
    parser.add_argument(
        "--cttp-config",
        type=Path,
        default=(
            root
            / "checkpoints/cttp/synth-u/resources/cttp/synth-u/model_configs.yaml"
        ),
    )
    parser.add_argument(
        "--cttp-checkpoint",
        type=Path,
        default=(
            root
            / "checkpoints/cttp/synth-u/resources/cttp/synth-u/clip_model_best.pth"
        ),
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Default: full test set")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=("guidance",),
        default="guidance",
        help="Steering operator: per-sample latent classifier guidance (eta sweep)",
    )
    parser.add_argument(
        "--strengths",
        default="2560",
        help="Comma-separated eta values in "
        "z_new = relu(z - eta*grad log p_target) (raw latent units)",
    )
    parser.add_argument(
        "--guidance-iters",
        type=int,
        default=1,
        help="Projected gradient-ascent iterations per in-window diffusion "
        "step (guidance mode only)",
    )
    parser.add_argument(
        "--guidance-rel-cap",
        type=float,
        default=1.0,
        help="Per-latent step cap in units of the classifier input_std "
        "(guidance mode only)",
    )
    parser.add_argument(
        "--guidance-max-step",
        type=float,
        default=0.5,
        help="Absolute per-latent step cap in raw latent units "
        "(guidance mode only)",
    )
    parser.add_argument(
        "--guidance-adaptive",
        type=float,
        default=0.0,
        help="Confidence-weighted guidance gamma: per-segment loss is scaled "
        "by (1 - p_target)^gamma so saturated segments (e.g. nothing at "
        "~98% accuracy) receive little perturbation (guidance modes only, "
        "0 disables)",
    )
    parser.add_argument(
        "--guidance-full-strength",
        default="",
        help="Comma-separated target shape names that keep full-strength "
        "guidance regardless of classifier confidence (shape-aware hybrid); "
        "other shapes use the adaptive weighting. Empty = pure adaptive "
        "(guidance modes only)",
    )
    parser.add_argument(
        "--windows",
        default="t5_45",
        help="Comma-separated SAE window names under results-root/models (default: t5_45)",
    )
    parser.add_argument("--cttp-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Reuse saved chunks")
    return parser


def _chunk_path(output_dir: Path, variant: str, start: int, end: int) -> Path:
    return output_dir / "chunks" / f"{variant}_{start:05d}_{end:05d}.npy"


def generate_all_variants(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    target_tuples: np.ndarray,
    wrappers: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    variant_keys: Sequence[str],
    *,
    extra_conditions: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Generate the paired variants chunk-by-chunk with resume support."""

    n_samples = len(embeddings)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        paths = {key: _chunk_path(output_dir, key, start, end) for key in variant_keys}
        if args.resume and all(path.is_file() for path in paths.values()):
            continue
        chunk_parts: dict[str, list[np.ndarray]] = {key: [] for key in variant_keys}
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)
            condition = torch.from_numpy(
                np.asarray(embeddings[batch_start:batch_end], dtype=np.float32)
            ).to(device)
            seed = args.seed + batch_start
            chunk_parts[PURE_KEY].append(
                generate_variant(model, condition, None, seed)
            )
            for variant in variant_keys:
                if variant == PURE_KEY:
                    continue
                names = target_tuples[batch_start:batch_end]
                name_to_index = {
                    name: index for index, name in enumerate(SHAPE_NAMES)
                }
                indices = np.asarray(
                    [[name_to_index[n] for n in row] for row in names],
                    dtype=np.int64,
                )
                wrappers[variant].set_targets(
                    torch.from_numpy(indices).to(device)
                )
                chunk_parts[variant].append(
                    generate_variant(model, condition, wrappers[variant], seed)
                )
        for key in variant_keys:
            chunk = np.concatenate(chunk_parts[key], axis=0)
            if chunk.shape[0] != end - start:
                raise ValueError(
                    f"chunk {key} [{start}, {end}) has {chunk.shape[0]} samples, "
                    f"expected {end - start}"
                )
            temporary = paths[key].with_suffix(".tmp.npy")
            np.save(temporary, chunk)
            os.replace(temporary, paths[key])
    model.verbalts.activation_transform = None
    return {
        key: np.concatenate(
            [
                np.load(
                    _chunk_path(output_dir, key, chunk_start,
                                min(chunk_start + CHUNK_SIZE, n_samples))
                )
                for chunk_start in range(0, n_samples, CHUNK_SIZE)
            ],
            axis=0,
        )
        for key in variant_keys
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    windows = tuple(
        window.strip() for window in args.windows.split(",") if window.strip()
    )
    if not windows:
        raise ValueError("--windows must list at least one window")
    unknown = [window for window in windows if window not in WINDOW_LABELS]
    if unknown:
        raise ValueError(
            f"unknown windows {unknown}; known: {sorted(WINDOW_LABELS)}"
        )
    strengths = [
        float(value) for value in args.strengths.split(",") if value.strip()
    ]
    if not strengths or any(s <= 0.0 for s in strengths):
        raise ValueError("--strengths must list positive eta values")
    steering_keys = [
        f"{window}_g{eta:g}"
        for window in windows
        for eta in strengths
    ]
    variant_keys = tuple([PURE_KEY, *steering_keys])
    labels = {PURE_KEY: "Pure VerbalTS"}
    for key in steering_keys:
        window, eta = key.rsplit("_g", 1)
        labels[key] = f"{WINDOW_LABELS[window]} (eta={eta})"
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Data and ground-truth tuples -------------------------------------
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
    target_tuples = shape_names  # (N, 3) of shape names in B/M/E order
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

    # --- Frozen evaluation CNN --------------------------------------------
    classifier_path = args.classifier_checkpoint.resolve()
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

    # --- VerbalTS and SAE guidance wrappers --------------------------------
    model_path = args.model_checkpoint.resolve()
    model, model_state_hash = load_verbalts(model_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    results_root = args.results_root.resolve()
    sae_by_window: dict[str, tuple[Any, tuple[int, int], Any]] = {}
    wrappers: dict[str, Any] = {}
    window_reports: dict[str, Any] = {}
    for window in windows:
        sae_path = results_root / "models" / window / "best.pt"
        sae, t_range, sae_checkpoint = load_sae_checkpoint(sae_path, device)
        sae_by_window[window] = (sae, t_range, sae_checkpoint)
        window_reports[window] = {
            "sae_checkpoint": str(sae_path),
            "sae_checkpoint_sha256": sha256_file(sae_path),
            "sae_state_sha256": sae_checkpoint.get(
                "state_dict_sha256", tensor_state_sha256(sae.state_dict())
            ),
            "t_range": list(t_range),
            "mode": args.mode,
        }
    for key in steering_keys:
        window, eta = key.rsplit("_g", 1)
        sae, t_range, _ = sae_by_window[window]
        classifier_path = (
            results_root / "classwise_classifiers" / window / "best.pt"
        )
        latent_clf, _ = load_classifier_checkpoint(classifier_path, device)
        latent_clf.eval()
        wrappers[key] = LatentClassifierGuidanceWrapper(
            sae,
            t_range,
            latent_clf,
            float(eta),
            rel_cap=args.guidance_rel_cap,
            max_step=args.guidance_max_step,
            iters=args.guidance_iters,
            adaptive_gamma=args.guidance_adaptive,
            full_strength_shapes=args.guidance_full_strength,
        ).to(device).eval()

    # --- Generation --------------------------------------------------------
    for wrapper in wrappers.values():
        wrapper.reset_audit()
    curves = generate_all_variants(
        model, embeddings, target_tuples, wrappers, args, output_dir, device,
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

    # --- Metrics -----------------------------------------------------------
    mse_reports = {
        key: compute_mse_report(curves[key], ground_truth, curves[PURE_KEY], args.seed)
        for key in variant_keys
    }
    cnn_reports = {
        key: compute_cnn_report(shape_names, peak_targets, valley_targets, predictions[key])
        for key in variant_keys
    }

    cttp_reports: dict[str, Any] = {}
    embedder, cttp_error = load_cttp_embedder(
        args.cttp_config.resolve(), args.cttp_checkpoint.resolve(), device, output_dir
    )
    if embedder is None:
        cttp_reports = {"status": "skipped", "reason": cttp_error}
    else:
        try:
            values = {}
            for key in variant_keys:
                values[key] = compute_cttp(
                    embedder, curves[key], captions, device, args.cttp_batch_size
                )
            cttp_reports = {"status": "computed", "cttp": values}
        except Exception as exc:  # noqa: BLE001
            cttp_reports = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    # --- Artifacts ---------------------------------------------------------
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

    hook_audits = {key: wrappers[key].audit() for key in steering_keys}
    method = "per-sample latent classifier guidance (capped projected gradient ascent)"
    steering_formula = (
        "z_new = relu(z - clamp(eta * grad_z mean_stage(-log p_target), "
        "min(rel_cap * sigma_d, max_step))) per in-window step"
    )
    summary = {
        "method": method,
        "steering_formula": steering_formula,
        "mode": args.mode,
        "strengths": [float(s) for s in strengths],
        "guidance_iters": args.guidance_iters,
        "guidance_rel_cap": args.guidance_rel_cap,
        "guidance_max_step": args.guidance_max_step,
        "guidance_adaptive": args.guidance_adaptive,
        "guidance_full_strength": args.guidance_full_strength,
        "sampler": "ddim",
        "seed": args.seed,
        "shared_random_stream_across_variants": True,
        "sample_range": [int(start), int(start + limit)],
        "n_samples": int(limit),
        "windows": window_reports,
        "mse": mse_reports,
        "cnn": cnn_reports,
        "cttp": cttp_reports,
        "evaluation_classifier_test": classifier_report.get("test", {}),
        "hook_audits": hook_audits,
        "audit": {
            "model_checkpoint": str(model_path),
            "model_checkpoint_sha256": sha256_file(model_path),
            "model_state_sha256": model_state_hash,
            "evaluation_classifier_checkpoint": str(classifier_path),
            "evaluation_classifier_checkpoint_sha256": sha256_file(classifier_path),
            "runtime": runtime_metadata(repo_root),
        },
    }
    write_json(output_dir / "summary.json", summary)

    # --- Comparison tables -------------------------------------------------
    rows = []
    for key in variant_keys:
        rows.append(
            {
                "variant": labels[key],
                "mse_overall": mse_reports[key]["overall_mse"],
                "mse_paired_delta_vs_pure": mse_reports[key]["paired_delta_vs_pure_mean"],
                "mse_delta_ci95_low": mse_reports[key]["paired_delta_vs_pure_ci95"][0],
                "mse_delta_ci95_high": mse_reports[key]["paired_delta_vs_pure_ci95"][1],
                "cttp": cttp_reports.get("cttp", {}).get(key, float("nan")),
                "segment_accuracy": cnn_reports[key]["segment_accuracy"],
                "whole_curve_exact_match": cnn_reports[key]["whole_curve_exact_match"],
                "non_nothing_segment_accuracy": cnn_reports[key][
                    "non_nothing_segment_accuracy"
                ],
                "mean_target_probability": cnn_reports[key]["mean_target_probability"],
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
    print(json.dumps({"mse": mse_reports, "cnn": cnn_reports, "cttp": cttp_reports},
                     indent=2, default=str))
    print(f"saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
