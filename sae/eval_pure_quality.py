"""Pure-generation quality evaluation for a VerbalTS checkpoint on 128 valid samples.

Compares against the previous synth-u-style model results already on disk
(results/electricity_v3/steering/valid128_synthustyle_1009823/pure*.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contsg.data.datasets.tsfresh_global import extract_global_features
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import compute_cnn_report, generate_variant
from sae.shapes import SHAPE_NAMES, captions_to_targets, classify_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="previous pure run (indices + pure_predictions.json)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dynamic-threshold", type=float, default=None,
        help="Optional per-sample DDIM x0 stabilization scale (for example 6.0)",
    )
    parser.add_argument(
        "--dynamic-threshold-quantile", type=float, default=0.995,
        help="Absolute x0 quantile used by --dynamic-threshold",
    )
    args = parser.parse_args()
    data, out, ref = args.data_root, args.output_dir, args.reference_dir
    out.mkdir(parents=True, exist_ok=True)
    reference_summary = json.loads((ref / "summary.json").read_text())
    indices = np.asarray(reference_summary["indices"])
    attrs = np.load(data / "valid_attrs_idx.npy")[indices]
    targets = attrs
    captions = np.load(data / "valid_text_caps.npy", allow_pickle=True).reshape(-1)[indices]
    names, peaks, valleys = captions_to_targets(captions)
    if not np.array_equal(names, np.asarray(SHAPE_NAMES)[targets]):
        raise ValueError("caption/attrs mismatch")
    embeddings = np.load(data / "valid_cap_emb.npy")[indices].astype(np.float32)
    truth = np.load(data / "valid_ts.npy")[indices, :, 0]
    device = torch.device(args.device)
    model, _ = load_verbalts(args.model_checkpoint, device)
    cnn = PeakValleyClassifier1D(segment_len=43).to(device)
    cnn.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device, weights_only=True))
    for network in (model, cnn):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)

    parts = []
    for start in range(0, len(indices), args.batch_size):
        end = min(start + args.batch_size, len(indices))
        with torch.no_grad():
            curves = generate_variant(
                model, torch.from_numpy(embeddings[start:end]).to(device),
                None, args.seed + start,
                dynamic_threshold=args.dynamic_threshold,
                dynamic_threshold_quantile=args.dynamic_threshold_quantile,
            )
        if not np.isfinite(curves).all():
            raise ValueError("non-finite generation")
        parts.append(curves)
    curves = np.concatenate(parts)
    np.save(out / "pure.npy", curves)
    predictions = classify_curves(cnn, curves, device)
    (out / "pure_predictions.json").write_text(json.dumps(predictions, indent=2))
    report = compute_cnn_report(names, peaks, valleys, predictions)

    # previous model results for comparison
    old_preds = json.loads((ref / "pure_predictions.json").read_text())
    old_report = compute_cnn_report(names, peaks, valleys, old_preds)

    print("=== segment accuracy (CNN) ===")
    print(f"  OLD synth-u-style model: {old_report['segment_accuracy']:.4f}")
    print(f"  NEW global-caption model: {report['segment_accuracy']:.4f}")
    print(f"  whole-curve exact: OLD {old_report['whole_curve_exact_match']:.4f} -> NEW {report['whole_curve_exact_match']:.4f}")
    print("=== per-shape recall ===")
    for shape in old_report["per_shape_recall"]:
        print(f"  {shape:<16} OLD {old_report['per_shape_recall'][shape]:.4f} -> "
              f"NEW {report['per_shape_recall'][shape]:.4f}")

    # roughness
    def roughness(x):
        return float(np.mean(np.abs(np.diff(x, axis=1))))
    train = np.load(data / "train_ts.npy")[:, :, 0]
    print("=== roughness (mean abs first-diff) ===")
    print(f"  train {roughness(train):.3f} | OLD {roughness(np.load(ref/'pure.npy')):.3f} | NEW {roughness(curves):.3f}")

    # point-level mismatch decomposition
    mse = float(np.mean((curves - truth) ** 2))
    old_mse = float(np.mean((np.load(ref / "pure.npy") - truth) ** 2))
    print("=== point-level MSE(GT, GEN) ===")
    print(f"  OLD {old_mse:.3f} | NEW {mse:.3f}")

    # global feature accuracy on generated curves (background fidelity check)
    print("=== global feature agreement with GT (background stats) ===")
    truth_feats = extract_global_features(truth, n_jobs=1)
    gen_feats = extract_global_features(curves, n_jobs=1)
    old_curve = np.load(ref / "pure.npy")
    old_feats = extract_global_features(old_curve, n_jobs=1)
    train_feats = np.load(data / "train_tsfresh_global.npy")
    std = train_feats.std(0)
    names_feat = ["trend_slope", "std", "mean_abs_change", "complexity", "autocorr_lag1"]
    for i, n in enumerate(names_feat):
        old_mae = float(np.mean(np.abs(old_feats[:, i] - truth_feats[:, i]) / std[i]))
        new_mae = float(np.mean(np.abs(gen_feats[:, i] - truth_feats[:, i]) / std[i]))
        print(f"  {n:<18} OLD {old_mae:.3f} -> NEW {new_mae:.3f}  (standardized MAE)")

    summary = dict(indices=indices.tolist(), cnn_report=report,
                   old_cnn_report=old_report,
                   sampling=dict(sampler="ddim", seed=args.seed,
                                 dynamic_threshold=args.dynamic_threshold,
                                 dynamic_threshold_quantile=args.dynamic_threshold_quantile),
                   roughness=dict(train=roughness(train), old=roughness(old_curve), new=roughness(curves)),
                   mse=dict(old=old_mse, new=mse),
                   global_feature_mae_old=(
                       (np.abs(old_feats - truth_feats) / std).mean(0).astype(float).tolist()
                   ),
                   global_feature_mae_new=(
                       (np.abs(gen_feats - truth_feats) / std).mean(0).astype(float).tolist()
                   ))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("PURE_QUALITY_EVALUATION_COMPLETED")


if __name__ == "__main__":
    main()
