"""Generation-level ablation controls.

Two modes answering "what does the SAE actually buy us?":

* ``hguidance`` — classifier guidance applied DIRECTLY on the raw 64-dim
  residual (no SAE encode/decode), eta grid.  If this works as well as the
  SAE-latent guidance, the SAE is not required for performance.
* ``sparse`` — SAE-latent guidance with the step truncated to top-k latents:
  ``global`` = fixed per-shape top-k of the classifier composite weight
  (answers "did we find feature neurons?"), ``dynamic`` = per-token top-k of
  the current gradient (answers "how many latents does one step need?").

Every variant shares the exact same sampling stream (paired DDIM, same seed)
and the same evaluation pipeline as the other steering studies.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.h_probe import HClassifier, train_classifier
from sae.eval_guidance import (
    PURE_KEY,
    captions_to_targets,
    classify_curves,
    compute_cnn_report,
    compute_mse_report,
    generate_variant,
)
from sae.shape_classifier import load_activation_cache, load_classifier_checkpoint
from sae.shapes import SHAPE_NAMES
from sae.steering import (
    HGuidanceWrapper,
    LatentClassifierGuidanceWrapper,
    load_sae_checkpoint,
)

# Registry side effect.
import contsg.models.verbalts  # noqa: F401, E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hguidance", "sparse"), required=True)
    parser.add_argument("--model-checkpoint", type=Path,
                        default=root / "artifacts/no_sae_pure_verbalts/verbalts.ckpt")
    parser.add_argument("--sae-checkpoint", type=Path,
                        default=root / "results/sae_retrain/models/t5_45/best.pt")
    parser.add_argument("--mlp-classifier", type=Path,
                        default=root / "results/sae_retrain/classwise_classifiers/t5_45/best.pt")
    parser.add_argument("--train-cache", type=Path,
                        default=root / "results/sae_retrain/activations_linig/train_layer1_t5_45.pt")
    parser.add_argument("--train-captions", type=Path,
                        default=root / "results/sae_retrain/activations_linig/train_caps_1024.npy")
    parser.add_argument("--valid-cache", type=Path,
                        default=root / "results/sae_retrain/activations_linig/valid_layer1_t5_45.pt")
    parser.add_argument("--valid-captions", type=Path,
                        default=root / "results/sae_retrain/activations_linig/valid_caps_512.npy")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=root / "results/sae_retrain/classwise_steering/evaluation_classifier/segment_cnn.pth")
    parser.add_argument("--data-root", type=Path, default=root / "datasets/synth-u")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/control_ablations")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.output_dir / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()

    captions = np.load(data_root / "test_text_caps.npy", allow_pickle=True).reshape(-1)
    shape_names_all, peak_all, valley_all = captions_to_targets(captions)
    total = len(captions)
    start = args.sample_offset
    limit = min(args.n_samples, total - start)
    indices = np.arange(start, start + limit)
    captions = captions[indices]
    shape_names = shape_names_all[indices]
    peak_targets = peak_all[indices]
    valley_targets = valley_all[indices]
    embeddings = np.asarray(
        np.load(data_root / "test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy",
                mmap_mode="r")[indices], dtype=np.float32)
    ground_truth = np.asarray(
        np.load(data_root / "test_ts.npy", mmap_mode="r")[indices, :, 0],
        dtype=np.float32)
    print(f"evaluating {limit} samples (offset {start}), mode={args.mode}")

    classifier = PeakValleyClassifier1D(segment_len=43).to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_checkpoint, map_location=device, weights_only=True),
        strict=True)
    classifier.eval()

    model, _ = load_verbalts(args.model_checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sae, t_range, _ = load_sae_checkpoint(args.sae_checkpoint, device)

    wrappers: dict[str, Any] = {}
    if args.mode == "hguidance":
        train_rows, _, train_labels, _ = load_activation_cache(
            args.train_cache, args.train_captions)
        valid_rows, _, valid_labels, _ = load_activation_cache(
            args.valid_cache, args.valid_captions)
        tokens = train_rows.shape[1]
        train_rows = train_rows.reshape(-1, train_rows.shape[-1])
        valid_rows = valid_rows.reshape(-1, valid_rows.shape[-1])
        train_labels = train_labels.repeat_interleave(tokens, dim=0)
        valid_labels = valid_labels.repeat_interleave(tokens, dim=0)
        h_clf, _ = train_classifier(
            train_rows, train_labels, valid_rows, valid_labels, device
        )
        for eta in (30.0, 100.0, 300.0):
            wrappers[f"hguidance_eta{eta:g}"] = HGuidanceWrapper(
                h_clf, t_range, eta=eta
            ).to(device).eval()
    else:  # sparse
        mlp, _ = load_classifier_checkpoint(args.mlp_classifier, device)
        mlp.eval()
        base = dict(
            eta=2560.0, adaptive_gamma=6.0,
            full_strength_shapes="double peaks,sag",
        )
        wrappers["sparse_dense"] = LatentClassifierGuidanceWrapper(
            sae, t_range, mlp, **base).to(device).eval()
        wrappers["sparse_dense_eta5120"] = LatentClassifierGuidanceWrapper(
            sae, t_range, mlp, **{**base, "eta": 5120.0}).to(device).eval()
        wrappers["sparse_global_k32"] = LatentClassifierGuidanceWrapper(
            sae, t_range, mlp, topk=32, topk_mode="global", **base
        ).to(device).eval()
        for k in (16, 32):
            wrappers[f"sparse_dynamic_k{k}"] = LatentClassifierGuidanceWrapper(
                sae, t_range, mlp, topk=k, topk_mode="dynamic", **base
            ).to(device).eval()
            wrappers[f"sparse_dynamic_k{k}_eta5120"] = \
                LatentClassifierGuidanceWrapper(
                    sae, t_range, mlp, topk=k, topk_mode="dynamic",
                    **{**base, "eta": 5120.0}
                ).to(device).eval()

    steering_keys = list(wrappers)
    variant_keys = tuple([PURE_KEY, *steering_keys])
    print("variants:", variant_keys)
    name_to_index = {name: index for index, name in enumerate(SHAPE_NAMES)}

    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def chunk_path(key: str, cs: int, ce: int) -> Path:
        return chunk_dir / f"{key}_{cs:05d}_{ce:05d}.npy"

    chunk_size = 250
    for chunk_start in range(0, limit, chunk_size):
        chunk_end = min(chunk_start + chunk_size, limit)
        paths = {key: chunk_path(key, chunk_start, chunk_end) for key in variant_keys}
        if args.resume and all(p.is_file() for p in paths.values()):
            continue
        parts: dict[str, list[np.ndarray]] = {key: [] for key in variant_keys}
        for batch_start in range(chunk_start, chunk_end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, chunk_end)
            condition = torch.from_numpy(
                np.asarray(embeddings[batch_start:batch_end], dtype=np.float32)
            ).to(device)
            seed = args.seed + batch_start
            parts[PURE_KEY].append(generate_variant(model, condition, None, seed))
            names = shape_names[batch_start:batch_end]
            indices_t = np.asarray(
                [[name_to_index[n] for n in row] for row in names], dtype=np.int64)
            for key in steering_keys:
                wrappers[key].set_targets(torch.from_numpy(indices_t).to(device))
                parts[key].append(
                    generate_variant(model, condition, wrappers[key], seed))
        for key in variant_keys:
            chunk = np.concatenate(parts[key], axis=0)
            temporary = paths[key].with_suffix(".tmp.npy")
            np.save(temporary, chunk)
            import os
            os.replace(temporary, paths[key])
    model.verbalts.activation_transform = None

    curves = {
        key: np.concatenate(
            [np.load(chunk_path(key, s, min(s + chunk_size, limit)))
             for s in range(0, limit, chunk_size)], axis=0)
        for key in variant_keys
    }
    predictions = {
        key: classify_curves(classifier, curves[key], device) for key in variant_keys
    }
    reports: dict[str, Any] = {}
    for key in variant_keys:
        reports[key] = {
            "cnn": compute_cnn_report(
                shape_names, peak_targets, valley_targets, predictions[key]),
            "mse": compute_mse_report(
                curves[key], ground_truth, curves[PURE_KEY], args.seed),
        }
        if key != PURE_KEY:
            reports[key]["audit"] = wrappers[key].audit()
    summary = {
        "mode": args.mode,
        "n_samples": limit,
        "sample_offset": start,
        "seed": args.seed,
        "variants": {
            key: {
                "segment_accuracy": reports[key]["cnn"]["segment_accuracy"],
                "whole_curve_exact_match": reports[key]["cnn"]["whole_curve_exact_match"],
                "overall_mse": reports[key]["mse"]["overall_mse"],
                "paired_delta_vs_pure_mean": reports[key]["mse"]["paired_delta_vs_pure_mean"],
            }
            for key in variant_keys
        },
        "reports": reports,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / "curves.npz", **{
        key: curves[key] for key in variant_keys
    })
    print("## summary")
    for key in variant_keys:
        entry = summary["variants"][key]
        print(f"{key:>24}: seg={entry['segment_accuracy']:.4f} "
              f"whole={entry['whole_curve_exact_match']:.4f} "
              f"mse={entry['overall_mse']:.4f} "
              f"dMSE={entry['paired_delta_vs_pure_mean']:+.4f}")
    print("## audits")
    for key in steering_keys:
        print(f"{key:>24}: {reports[key]['audit']}")


if __name__ == "__main__":
    main()
