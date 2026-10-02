"""Paired morphology steering evaluation for verbalts/diffusets/bridge.

Extension of ``sae.eval_morph_steering`` that loads any registered generator,
attaches the SAE/guidance wrapper to the model-specific activation hook, and
reuses the identical metric pipeline (CNN ACCR / MSE / global categories).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contsg.data.datasets.tsfresh_global import FEATURE_NAMES, extract_global_features, fit_caption_thresholds
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_generator
from sae.eval_guidance import compute_cnn_report, compute_mse_report
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


def core_of(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    return {
        "verbalts": getattr(model, "verbalts", model),
        "diffusets": getattr(model, "unet", model),
        "bridge": getattr(model, "model", model),
    }[model_name]


def generate_variant_any(
    model: torch.nn.Module,
    model_name: str,
    condition: torch.Tensor,
    wrapper: torch.nn.Module | None,
    seed: int,
    sampler: str = "ddim",
    bridge_example_ts: torch.Tensor | None = None,
    transform_block_index: int = 2,
) -> np.ndarray:
    """Generate one curve per condition under an optional SAE wrapper."""
    core = core_of(model, model_name)
    # DiffuSETS: keep the hook at the same down-block index used for
    # activation collection (default index=2: L=2 tokens, C=32).
    if model_name == "diffusets" and hasattr(core, "transform_block_index"):
        core.transform_block_index = transform_block_index
    core.activation_transform = wrapper
    torch.manual_seed(seed)
    if condition.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if model_name == "bridge":
        generated = model.generate(
            condition, n_samples=1, sampler=sampler,
            bridge_example_ts=bridge_example_ts,
        )
        # Bridge/diffusets return (B, n_samples, L, C); verbalts returns
        # (n_samples, B, L, C).
        return generated[:, 0, :, 0].detach().cpu().float().numpy()
    if model_name == "diffusets":
        generated = model.generate(condition, n_samples=1, sampler=sampler)
        return generated[:, 0, :, 0].detach().cpu().float().numpy()
    generated = model.generate(condition, n_samples=1, sampler=sampler)
    return generated[0, :, :, 0].detach().cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["auto", "verbalts", "diffusets", "bridge"], default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mlp-path", type=Path, default=None,
        help="Optional explicit MLP checkpoint (defaults to classwise_classifiers/t5_45/best.pt "
             "under --results-root)",
    )
    parser.add_argument(
        "--sae-path", type=Path, default=None,
        help="Optional explicit SAE checkpoint (defaults to models/t5_45/best.pt under --results-root)",
    )
    parser.add_argument("--n-samples", type=int, default=128, help="0 evaluates the entire split")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--transform-block-index", type=int, default=2,
        help="DiffuSETS down-block index for the steering hook (must match "
             "the position used to train the SAE/MLP)",
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
    text_caps_path = data / f"{split}_text_caps.npy"
    captions = np.load(text_caps_path, allow_pickle=True).reshape(-1)[indices]
    names, peaks, valleys = captions_to_targets(captions)
    if not np.array_equal(names, np.asarray(SHAPE_NAMES)[targets]):
        raise ValueError("caption/attrs mismatch")
    embeddings = np.load(data / f"{split}_cap_emb.npy")[indices].astype(np.float32)
    truth = np.load(data / f"{split}_ts.npy")[indices, :, 0]
    train_features = np.load(data / "train_tsfresh_global.npy")
    thresholds = fit_caption_thresholds(train_features)
    truth_features = np.load(data / f"{split}_tsfresh_global.npy")[indices]
    truth_levels = global_levels(truth_features, thresholds)
    device = torch.device(args.device)
    model, model_name = load_generator(args.model_checkpoint, device)
    if args.model_type != "auto" and args.model_type != model_name:
        raise ValueError(f"checkpoint model is {model_name}, but --model-type={args.model_type}")
    print(f"MODEL_TYPE={model_name}", flush=True)
    sae_path = args.sae_path or (root / "models/t5_45/best.pt")
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
    variants = json.loads(args.config_file.read_text()) if args.config_file else [
        dict(name="pure"), dict(name="sae"),
        dict(name="grad_c2_s1", topk=32, eta=10240, gamma=6, rel_cap=2.0, max_step=1.0),
        dict(name="grad_c4_s2", topk=32, eta=10240, gamma=6, rel_cap=4.0, max_step=2.0),
        dict(name="applied_c2_s1", topk=32, eta=5120, gamma=6, rel_cap=2.0, max_step=1.0,
             selection_score="applied", preserve_residual=True),
    ]
    if variants[:2] != [dict(name="pure"), dict(name="sae")]:
        raise ValueError("custom configurations must start with pure and sae")
    # Bridge prototype source: first train batch (same protocol as contsg evaluator).
    bridge_example_ts = None
    if model_name == "bridge":
        train_ts = np.load(data / "train_ts.npy")
        base = torch.from_numpy(train_ts[:args.batch_size]).float().to(device)  # (B, L, C)
        bridge_example_ts = base
    report = dict(split=split, indices=indices.tolist(), seed=args.seed,
                  model_type=model_name,
                  batch_size=args.batch_size, paired_noise=True,
                  sampling={"sampler": "ddim"},
                  variants={},
                  checkpoints={str(p): sha256_file(p) for p in
                               (args.model_checkpoint, sae_path, mlp_path, cnn_path)},
                  data_sha256={name: sha256_file(data / name) for name in
                               (f"{split}_ts.npy", f"{split}_attrs_idx.npy", f"{split}_text_caps.npy",
                                f"{split}_cap_emb.npy", f"{split}_tsfresh_global.npy", "train_tsfresh_global.npy")})
    pure = None
    pure_features = None
    for config in variants:
        name = config["name"]
        wrapper = None
        if name == "sae":
            wrapper = TimestepSAEWrapper(sae, window)
        elif name != "pure":
            wrapper = LatentClassifierGuidanceWrapper(
                sae, window, mlp, eta=config["eta"], adaptive_gamma=config.get("gamma", 6),
                topk=config.get("topk", 32), topk_mode=config.get("topk_mode", "dynamic"),
                rel_cap=config.get("rel_cap", 1.), max_step=config.get("max_step", .5),
                objective_head_indices=config.get("objective_head_indices"),
                objective_head_weights=config.get("objective_head_weights"),
                objective_class_weights=config.get("objective_class_weights"),
                guidance_t_range=config.get("guidance_t_range"),
                active_only=config.get("active_only", False),
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
            with torch.no_grad():
                curves = generate_variant_any(
                    model, model_name,
                    torch.from_numpy(embeddings[start:end]).to(device),
                    wrapper,
                    args.seed + start,
                    bridge_example_ts=(
                        bridge_example_ts[: end - start] if bridge_example_ts is not None else None
                    ),
                    transform_block_index=args.transform_block_index,
                )
            if not np.isfinite(curves).all():
                raise ValueError(f"non-finite generation in {name}")
            parts.append(curves)
        curves = np.concatenate(parts)
        np.save(out / f"{name}.npy", curves)
        predictions = classify_curves(cnn, curves, device)
        write_json(out / f"{name}_predictions.json", predictions)
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
    core_of(model, model_name).activation_transform = None
    print("STEERING_EVALUATION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
