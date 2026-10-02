"""Train the paper-style lightweight ``F_x(s)`` classifier on SAE latents."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sae.shape_classifier import (
    CLASSIFIER_SCHEMA_VERSION,
    SAEClassClassifier,
    classifier_metrics,
    encode_sae_latents,
    load_activation_cache,
    weighted_classification_loss,
)
from sae.provenance import runtime_metadata, sha256_file, tensor_state_sha256, write_json
from sae.steering import load_sae_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--valid-cache", type=Path, required=True)
    train_labels = parser.add_mutually_exclusive_group(required=True)
    train_labels.add_argument("--train-captions", type=Path)
    train_labels.add_argument("--train-attrs", type=Path)
    valid_labels = parser.add_mutually_exclusive_group(required=True)
    valid_labels.add_argument("--valid-captions", type=Path)
    valid_labels.add_argument("--valid-attrs", type=Path)
    parser.add_argument(
        "--train-soft-attrs", type=Path, default=None,
        help="optional (samples, 3, 4) soft target probabilities that "
             "REPLACE the hard labels for the training loss (metrics still "
             "use the hard labels above); enables CNN-prob distillation"
    )
    parser.add_argument(
        "--valid-soft-attrs", type=Path, default=None,
        help="optional soft targets for validation loss reporting only"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-heads", type=int, default=3,
        help="number of classifier heads (3 for three-stage sequences, 1 for single-beat)"
    )
    parser.add_argument(
        "--num-shape-classes", type=int, default=None,
        help="classes per head (4 = nothing + 3 coarse shapes; 6 = nothing + 5 fine shapes)"
    )
    parser.add_argument(
        "--skip-encode", action="store_true",
        help="use the cache rows directly as MLP inputs instead of "
             "re-encoding them with the SAE; pairs with a --direct-latents "
             "generated cache so training sees the exact tensor the "
             "steering harness exposes to the MLP"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--selection-metric", choices=("auto", "accr"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _class_weights(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    weights = torch.empty(labels.shape[1], num_classes, dtype=torch.float32)
    for stage in range(labels.shape[1]):
        counts = torch.bincount(labels[:, stage], minlength=num_classes).float()
        weights[stage] = counts.sum() / counts.clamp_min(1) / num_classes
    return weights


def _load_soft_labels(soft_path: Path, sample_count: int, num_heads: int = 3) -> torch.Tensor:
    """Load (samples, num_heads, 4) soft target probabilities from a .npy file."""

    raw = np.load(soft_path)
    if raw.ndim != 3 or raw.shape[1] != num_heads or raw.shape[2] != 4:
        raise ValueError(
            f"soft labels must have shape (samples, {num_heads}, 4), got {raw.shape}"
        )
    if raw.shape[0] != sample_count:
        raise ValueError(
            f"soft labels {raw.shape[0]} do not match {sample_count} cached samples"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("soft labels contain non-finite values")
    return torch.from_numpy(raw.astype(np.float32, copy=False))


def _soft_classification_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-stage soft cross-entropy, weighted by the argmax class weight."""

    losses = []
    for stage in range(logits.shape[1]):
        log_prob = logits[:, stage].log_softmax(-1)
        terms = -(soft_targets[:, stage] * log_prob).sum(-1)
        hard = soft_targets[:, stage].argmax(-1)
        terms = terms * class_weights[stage][hard]
        losses.append(terms.mean())
    return torch.stack(losses).mean()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.num_heads < 1:
        raise ValueError("num-heads must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sae_path = args.sae_checkpoint.resolve()
    sae, t_range, sae_checkpoint = load_sae_checkpoint(sae_path, device=device)
    if args.num_shape_classes is not None and args.num_shape_classes not in (4, 6):
        raise ValueError("num-shape-classes must be 4 or 6")
    train_label_path = args.train_attrs or args.train_captions
    valid_label_path = args.valid_attrs or args.valid_captions
    train_samples, _, train_labels, train_payload = load_activation_cache(
        args.train_cache, train_label_path, labels_from_attrs=args.train_attrs is not None,
        num_heads=args.num_heads, num_shape_classes=args.num_shape_classes
    )
    valid_samples, _, valid_labels, valid_payload = load_activation_cache(
        args.valid_cache, valid_label_path, labels_from_attrs=args.valid_attrs is not None,
        num_heads=args.num_heads, num_shape_classes=args.num_shape_classes
    )
    train_soft = (
        _load_soft_labels(args.train_soft_attrs, train_samples.shape[0], args.num_heads)
        if args.train_soft_attrs is not None else None
    )
    valid_soft = (
        _load_soft_labels(args.valid_soft_attrs, valid_samples.shape[0], args.num_heads)
        if args.valid_soft_attrs is not None else None
    )
    if train_soft is not None:
        mismatch = int(train_soft.argmax(-1).ne(train_labels).float().mean() * 100)
        print(f"soft-target argmax disagrees with hard labels on {mismatch}% of rows")
    if tuple(train_payload["t_range"]) != t_range or tuple(valid_payload["t_range"]) != t_range:
        raise ValueError("SAE and activation-cache timestep ranges do not match")

    if args.skip_encode:
        if train_samples.shape[-1] != sae.latent_dim or valid_samples.shape[-1] != sae.latent_dim:
            raise ValueError(
                "--skip-encode requires cache feature dim == sae.latent_dim "
                f"({sae.latent_dim}), got train={train_samples.shape[-1]}, "
                f"valid={valid_samples.shape[-1]}"
            )
        train_features = train_samples.float()
        valid_features = valid_samples.float()
    else:
        train_features = encode_sae_latents(
            train_samples, sae, device, args.encode_batch_size
        )
        valid_features = encode_sae_latents(
            valid_samples, sae, device, args.encode_batch_size
        )
    input_mean = train_features.float().mean(dim=(0, 1))
    input_std = train_features.float().std(dim=(0, 1)).clamp_min(1e-6)
    classifier = SAEClassClassifier(
        sae.latent_dim, train_features.shape[1], args.hidden_dim, args.dropout,
        input_mean, input_std, num_heads=args.num_heads,
        num_shape_classes=args.num_shape_classes
    ).to(device)
    weights = _class_weights(train_labels, classifier.num_shape_classes).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    if train_soft is not None:
        loader = DataLoader(
            TensorDataset(train_features, train_labels, train_soft),
            batch_size=args.batch_size, shuffle=True, generator=generator,
        )
    else:
        loader = DataLoader(
            TensorDataset(train_features, train_labels), batch_size=args.batch_size,
            shuffle=True, generator=generator,
        )
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    best_accuracy = -1.0
    best_epoch = 0
    no_improvement = 0
    history = []
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        loss_sum = 0.0
        rows = 0
        for batch in loader:
            features = batch[0].to(device, dtype=torch.float32)
            labels = batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            if train_soft is not None:
                soft = batch[2].to(device, dtype=torch.float32)
                loss = _soft_classification_loss(
                    classifier(features), soft, weights
                )
            else:
                loss = weighted_classification_loss(
                    classifier(features), labels, weights
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(features)
            rows += len(features)
        scheduler.step()
        valid_metrics = classifier_metrics(
            classifier, valid_features, valid_labels, device, args.batch_size
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / rows,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "valid_mean_stage_accuracy": valid_metrics["mean_stage_accuracy"],
            "valid_joint_three_stage_accuracy": valid_metrics["joint_three_stage_accuracy"],
        }
        history.append(record)
        accuracy = float(valid_metrics["accr"] if args.num_heads == 4 or args.selection_metric == "accr" else valid_metrics["mean_stage_accuracy"])
        print(
            f"epoch {epoch:03d}: loss={record['train_loss']:.6f}, "
            f"valid_stage_acc={accuracy:.4f}, "
            f"valid_joint_acc={record['valid_joint_three_stage_accuracy']:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            no_improvement = 0
            torch.save({
                "schema_version": CLASSIFIER_SCHEMA_VERSION,
                "state_dict": {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
                "latent_dim": sae.latent_dim,
                "tokens_per_sample": train_features.shape[1],
                "num_heads": int(args.num_heads),
                "num_shape_classes": int(classifier.num_shape_classes),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "input_mean": input_mean,
                "input_std": input_std,
                "t_range": list(t_range),
                "sae_checkpoint": str(sae_path),
                "sae_checkpoint_sha256": sha256_file(sae_path),
                "train_cache": str(args.train_cache.resolve()),
                "train_cache_sha256": sha256_file(args.train_cache.resolve()),
                "valid_cache": str(args.valid_cache.resolve()),
                "valid_cache_sha256": sha256_file(args.valid_cache.resolve()),
                "skip_encode": bool(args.skip_encode),
                "best_epoch": best_epoch,
            }, checkpoint_path)
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(best["state_dict"], strict=True)
    classifier.to(device).eval()
    report = {
        "classifier": "SAEClassClassifier",
        "target": f"{args.num_heads} positions x four semantic shape classes",
        "input": f"position-aware {train_features.shape[1]} x {sae.latent_dim} token-wise SAE latent tensor",
        "selection_metric": "accr" if args.num_heads == 4 or args.selection_metric == "accr" else "mean_stage_accuracy",
        "skip_encode": bool(args.skip_encode),
        "t_range": list(t_range),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "soft_targets": {
            "train": str(args.train_soft_attrs.resolve()) if args.train_soft_attrs else None,
            "valid": str(args.valid_soft_attrs.resolve()) if args.valid_soft_attrs else None,
        },
        "train": classifier_metrics(
            classifier, train_features, train_labels, device, args.batch_size
        ),
        "valid": classifier_metrics(
            classifier, valid_features, valid_labels, device, args.batch_size
        ),
        "history": history,
        "audit": {
            "sae_checkpoint": str(sae_path),
            "sae_checkpoint_sha256": sha256_file(sae_path),
            "sae_state_sha256": sae_checkpoint.get(
                "state_dict_sha256", tensor_state_sha256(sae.state_dict())
            ),
            "train_cache": str(args.train_cache.resolve()),
            "train_cache_sha256": sha256_file(args.train_cache.resolve()),
            "valid_cache": str(args.valid_cache.resolve()),
            "valid_cache_sha256": sha256_file(args.valid_cache.resolve()),
            "skip_encode": bool(args.skip_encode),
            "classifier_checkpoint": str(checkpoint_path.resolve()),
            "classifier_checkpoint_sha256": sha256_file(checkpoint_path),
            "runtime": runtime_metadata(repo_root),
            "seed": args.seed,
        },
    }
    write_json(output_dir / "metrics.json", report)
    print(f"saved latent classifier: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()