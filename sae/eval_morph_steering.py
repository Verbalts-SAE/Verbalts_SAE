"""Paired morphology steering evaluation on validation or explicitly selected test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contsg.data.datasets.tsfresh_global import FEATURE_NAMES, extract_global_features, fit_caption_thresholds
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import compute_cnn_report, compute_mse_report, generate_variant
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import load_classifier_checkpoint
from sae.shapes import SHAPE_NAMES, captions_to_targets, classify_curves
from sae.steering import LatentClassifierGuidanceWrapper, TimestepSAEWrapper, load_sae_checkpoint


def global_levels(features: np.ndarray, thresholds: dict) -> np.ndarray:
    bounds = np.asarray([thresholds[name] for name in FEATURE_NAMES])
    return np.where(features < bounds[:, 0], 0, np.where(features > bounds[:, 1], 2, 1))


def evaluation_indices(total, count, offset=0):
    count = total if count == 0 else count
    if not 0 < count <= total or offset < 0 or offset + count > total:
        raise ValueError("sample offset/count exceeds selected split")
    return np.random.default_rng(2026).permutation(total)[offset:offset + count]


def configurations() -> list[dict]:
    variants = [dict(name="pure"), dict(name="sae")]
    for k, eta, gamma in [(0, 5120, 6), (32, 1280, 6), (32, 5120, 6),
                          (32, 20480, 6), (16, 5120, 6), (64, 5120, 6),
                          (32, 5120, 0), (32, 5120, 3),
                          (8, 5120, 6), (128, 5120, 6),
                          (32, 8192, 6), (32, 5120, 9)]:
        variants.append(dict(name=f"{'dense' if k == 0 else 'dynamic'}_k{k}_e{eta}_g{gamma}",
                             topk=k, eta=eta, gamma=gamma))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mlp-path", type=Path, default=None,
        help="Optional explicit MLP checkpoint (defaults to classwise_classifiers/t5_45/best.pt "
             "under --results-root); used for on-policy trained classifiers",
    )
    parser.add_argument("--n-samples", type=int, default=128, help="0 evaluates the entire split")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--cap-emb-file", type=Path, default=None,
        help="Optional explicit path to the split caption-embedding file "
        "(overrides {split}_cap_emb.npy under --data-root)",
    )
    parser.add_argument(
        "--text-caps-file", type=Path, default=None,
        help="Optional explicit path to the split text-caption file "
        "(overrides {split}_text_caps.npy under --data-root)",
    )
    args = parser.parse_args()
    data, root, out = args.data_root, args.results_root, args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty to prevent mixing runs")
    out.mkdir(parents=True, exist_ok=True)
    split = args.split
    attrs = np.load(data / f"{split}_attrs_idx.npy")
    if args.batch_size < 1:
        raise ValueError("invalid batch size")
    indices = evaluation_indices(len(attrs), args.n_samples, args.sample_offset)
    targets = attrs[indices]
    cap_emb_path = args.cap_emb_file or (data / f"{split}_cap_emb.npy")
    text_caps_path = args.text_caps_file or (data / f"{split}_text_caps.npy")
    captions = np.load(text_caps_path, allow_pickle=True).reshape(-1)[indices]
    names, peaks, valleys = captions_to_targets(captions)
    if not np.array_equal(names, np.asarray(SHAPE_NAMES)[targets]):
        raise ValueError("caption/attrs mismatch")
    embeddings = np.load(cap_emb_path)[indices].astype(np.float32)
    truth = np.load(data / f"{split}_ts.npy")[indices, :, 0]
    train_features = np.load(data / "train_tsfresh_global.npy")
    thresholds = fit_caption_thresholds(train_features)
    truth_features = np.load(data / f"{split}_tsfresh_global.npy")[indices]
    truth_levels = global_levels(truth_features, thresholds)
    device = torch.device(args.device)
    model, _ = load_verbalts(args.model_checkpoint, device)
    sae_path = root / "models/t5_45/best.pt"
    mlp_path = args.mlp_path or (root / "classwise_classifiers/t5_45/best.pt")
    cnn_path = root / "cnn/segment_cnn.pth"
    sae, window, _ = load_sae_checkpoint(sae_path, device)
    mlp, _ = load_classifier_checkpoint(mlp_path, device)
    cnn = PeakValleyClassifier1D(segment_len=43).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    for network in (model, sae, mlp, cnn):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    variants = configurations()
    if args.diagnose:
        base = dict(topk=32, eta=5120, gamma=6)
        variants = [dict(name="pure"), dict(name="sae")]
        for name, overrides in [
            ("reference", {}), ("middle", dict(objective_head_indices=[1])),
            ("late", dict(guidance_t_range=(5, 20))),
            ("early", dict(guidance_t_range=(30, 45))),
            ("cap_small", dict(rel_cap=.25, max_step=.125)),
            ("cap_large", dict(rel_cap=4., max_step=2.)),
            ("active_only", dict(active_only=True)),
        ]:
            variants.append(dict(name=name, **base, **overrides))
    if args.smoke:
        variants = variants[:3] + [variants[4]]
    if args.config_file:
        variants = json.loads(args.config_file.read_text())
        if variants[:2] != [dict(name="pure"), dict(name="sae")]:
            raise ValueError("custom configurations must start with pure and sae")
        if len({v["name"] for v in variants}) != len(variants):
            raise ValueError("configuration names must be unique")
    report = dict(split=split, indices=indices.tolist(), seed=args.seed,
                  source_sha256={str(p): sha256_file(p) for p in
                                 (Path(__file__).resolve(), Path(__file__).with_name("steering.py").resolve(),
                                  Path(__file__).with_name("eval_guidance.py").resolve())},
                  batch_size=args.batch_size, paired_noise=True,
                  sampling={
                      "sampler": "ddim",
                      "dynamic_threshold": args.dynamic_threshold,
                      "dynamic_threshold_quantile": args.dynamic_threshold_quantile,
                  },
                  variants={},
                  checkpoints={str(p): sha256_file(p) for p in
                               (args.model_checkpoint, sae_path, mlp_path, cnn_path)},
                  data_sha256={name: sha256_file(data / name) for name in
                               (f"{split}_ts.npy", f"{split}_attrs_idx.npy", f"{split}_text_caps.npy",
                                f"{split}_cap_emb.npy", f"{split}_tsfresh_global.npy", "train_tsfresh_global.npy")},
                  data_sha256_explicit={str(p): sha256_file(p) for p in (cap_emb_path, text_caps_path)})
    pure = None
    pure_features = None
    for config in variants:
        name = config["name"]
        wrapper = None
        if name == "sae":
            wrapper = TimestepSAEWrapper(sae, window)
        elif name != "pure":
            wrapper = LatentClassifierGuidanceWrapper(
                sae, window, mlp, eta=config["eta"], adaptive_gamma=config["gamma"],
                topk=config["topk"], topk_mode=config.get("topk_mode", "dynamic"),
                rel_cap=config.get("rel_cap", 1.), max_step=config.get("max_step", .5),
                objective_head_indices=config.get("objective_head_indices"),
                objective_head_weights=config.get("objective_head_weights"),
                objective_class_weights=config.get("objective_class_weights"),
                guidance_t_range=config.get("guidance_t_range"),
                active_only=config.get("active_only", False), diagnostic_trace=args.diagnose,
                selection_score=config.get("selection_score", "gradient"),
                preserve_residual=config.get("preserve_residual", False),
                iters=config.get("iters", 1),
                full_strength_shapes=config.get("full_strength_shapes", ""),
                batch_invariant=config.get("batch_invariant", False),
            )
        if wrapper is not None:
            wrapper.to(device).eval()
        parts = []
        for start in range(0, len(indices), args.batch_size):
            end = min(start + args.batch_size, len(indices))
            if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                wrapper.set_targets(torch.from_numpy(targets[start:end]).long().to(device))
            trace_start = len(wrapper.trace_records) if isinstance(wrapper, LatentClassifierGuidanceWrapper) else 0
            with torch.no_grad():
                curves = generate_variant(
                    model,
                    torch.from_numpy(embeddings[start:end]).to(device),
                    wrapper,
                    args.seed + start,
                    dynamic_threshold=args.dynamic_threshold,
                    dynamic_threshold_quantile=args.dynamic_threshold_quantile,
                )
            if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                for row in wrapper.trace_records[trace_start:]:
                    position = start + row["batch_row"]
                    row.update(sample_position=position, sample_index=int(indices[position]),
                               batch_start=start, generation_seed=args.seed + start)
            if not np.isfinite(curves).all():
                raise ValueError(f"non-finite generation in {name}")
            parts.append(curves)
        curves = np.concatenate(parts)
        np.save(out / f"{name}.npy", curves)
        predictions = classify_curves(cnn, curves, device)
        write_json(out / f"{name}_predictions.json", predictions)
        if isinstance(wrapper, LatentClassifierGuidanceWrapper) and args.diagnose:
            write_json(out / f"{name}_trace.json", wrapper.trace_records)
        features = extract_global_features(curves, n_jobs=1)
        np.save(out / f"{name}_global.npy", features)
        if name == "pure":
            pure, pure_features = curves, features
        assert pure is not None and pure_features is not None
        levels = global_levels(features, thresholds)
        metrics = dict(config=config,
                       roughness=float(np.abs(np.diff(curves, axis=1)).mean()),
                       curvature=float(np.abs(np.diff(curves, n=2, axis=1)).mean()),
                       truth_roughness=float(np.abs(np.diff(truth, axis=1)).mean()),
                       truth_curvature=float(np.abs(np.diff(truth, n=2, axis=1)).mean()),
                       absolute_max=float(np.abs(curves).max()),
                       cnn=compute_cnn_report(names, peaks, valleys, predictions),
                       mse=compute_mse_report(curves, truth, pure, args.seed),
                       global_category_accuracy=float((levels == truth_levels).mean()),
                       global_category_per_feature=dict(zip(FEATURE_NAMES, (levels == truth_levels).mean(0).tolist())),
                       global_preservation_vs_pure=float((levels == global_levels(pure_features, thresholds)).mean()),
                       global_standardized_mae=float((np.abs(features - truth_features) /
                                                     np.maximum(train_features.std(0), 1e-8)).mean()),
                       audit={} if wrapper is None else wrapper.audit())
        report["variants"][name] = metrics
        write_json(out / "summary.json", report)
        print(name, json.dumps(metrics), flush=True)
    model.verbalts.activation_transform = None
    print("STEERING_EVALUATION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()