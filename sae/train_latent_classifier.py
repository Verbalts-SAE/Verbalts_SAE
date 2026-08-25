"""Train the paper-style lightweight ``F_x(s)`` classifier on SAE latents."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sae.latent_classifier import (
    CLASSIFIER_SCHEMA_VERSION,
    SAEClassClassifier,
    classifier_metrics,
    encode_sae_latents,
    load_activation_cache,
    weighted_classification_loss,
)
from sae.provenance import runtime_metadata, sha256_file, tensor_state_sha256, write_json
from sae.wrappers import load_sae_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--valid-cache", type=Path, required=True)
    parser.add_argument("--train-captions", type=Path, required=True)
    parser.add_argument("--valid-captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
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


def _class_weights(labels: torch.Tensor) -> torch.Tensor:
    weights = torch.empty(labels.shape[1], 4, dtype=torch.float32)
    for stage in range(labels.shape[1]):
        counts = torch.bincount(labels[:, stage], minlength=4).float()
        weights[stage] = counts.sum() / counts.clamp_min(1) / len(counts)
    return weights


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
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
    train_samples, _, train_labels, train_payload = load_activation_cache(
        args.train_cache, args.train_captions
    )
    valid_samples, _, valid_labels, valid_payload = load_activation_cache(
        args.valid_cache, args.valid_captions
    )
    if tuple(train_payload["t_range"]) != t_range or tuple(valid_payload["t_range"]) != t_range:
        raise ValueError("SAE and activation-cache timestep ranges do not match")

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
        input_mean, input_std
    ).to(device)
    weights = _class_weights(train_labels).to(device)
    generator = torch.Generator().manual_seed(args.seed)
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
        for features, labels in loader:
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_classification_loss(classifier(features), labels, weights)
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
        accuracy = float(valid_metrics["mean_stage_accuracy"])
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
        "target": "three positions x four semantic shape classes",
        "input": "position-aware 47 x 512 token-wise SAE latent tensor",
        "t_range": list(t_range),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
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