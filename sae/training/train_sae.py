"""Train a pure Top-K SAE on one reproducible activation cache."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from contsg.models.sae_module import TopKSparseAutoencoder
from sae.provenance import runtime_metadata, sha256_file, tensor_state_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--valid-activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t-range", nargs=2, type=int, metavar=("MIN", "MAX"), required=True)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--topk-k", type=int, default=48)
    parser.add_argument("--aux-lambda", type=float, default=0.01)
    parser.add_argument("--dead-tolerance", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--resample-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    decoder_group = parser.add_mutually_exclusive_group()
    decoder_group.add_argument(
        "--normalize-decoder", dest="normalize_decoder", action="store_true"
    )
    decoder_group.add_argument(
        "--no-normalize-decoder", dest="normalize_decoder", action="store_false"
    )
    parser.set_defaults(normalize_decoder=True)
    return parser.parse_args()


def load_activations(path: Path, input_dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"activation cache not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        activations = loaded.get("activations")
        metadata = {key: value for key, value in loaded.items() if key != "activations"}
    else:
        activations = loaded
        metadata = {}
    if not isinstance(activations, torch.Tensor) or activations.ndim != 2:
        raise ValueError("activation cache must contain a 2D 'activations' tensor")
    if activations.shape[1] != input_dim:
        raise ValueError(
            f"activation feature dim is {activations.shape[1]}, expected {input_dim}"
        )
    return activations.float(), metadata


def main() -> None:
    args = parse_args()
    if args.t_range[0] > args.t_range[1]:
        raise ValueError("timestep range minimum must not exceed maximum")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    activations, metadata = load_activations(args.activations, args.input_dim)
    valid_activations, valid_metadata = load_activations(
        args.valid_activations, args.input_dim
    )
    cached_range = metadata.get("t_range")
    if cached_range is not None and tuple(cached_range) != tuple(args.t_range):
        raise ValueError(
            f"activation cache range {tuple(cached_range)} does not match {tuple(args.t_range)}"
        )
    valid_cached_range = valid_metadata.get("t_range")
    if valid_cached_range is not None and tuple(valid_cached_range) != tuple(args.t_range):
        raise ValueError(
            f"validation cache range {tuple(valid_cached_range)} does not match {tuple(args.t_range)}"
        )
    dataset = TensorDataset(activations)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(
        TensorDataset(valid_activations), batch_size=args.batch_size, shuffle=False
    )

    model = TopKSparseAutoencoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        topk_k=args.topk_k,
        aux_lambda=args.aux_lambda,
        normalize_decoder=args.normalize_decoder,
        dead_tolerance=args.dead_tolerance,
    )
    model.update_statistics(activations)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def warmup(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step) / float(max(args.warmup_steps, 1))
        return 1.0

    scheduler = LambdaLR(optimizer, warmup)
    config = {
        "input_dim": args.input_dim,
        "latent_dim": args.latent_dim,
        "topk_k": args.topk_k,
        "aux_lambda": args.aux_lambda,
        "dead_tolerance": args.dead_tolerance,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "resample_interval": args.resample_interval,
        "normalize_decoder": args.normalize_decoder,
        "t_range": list(args.t_range),
        "seed": args.seed,
        "train_activation_path": str(args.activations.resolve()),
        "train_activation_sha256": sha256_file(args.activations),
        "valid_activation_path": str(args.valid_activations.resolve()),
        "valid_activation_sha256": sha256_file(args.valid_activations),
        "train_activation_source": metadata.get("source"),
        "valid_activation_source": valid_metadata.get("source"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[2]
    training_log = []
    global_step = 0
    best_valid_recon = float("inf")
    best_epoch = 0

    def make_checkpoint(epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        state_dict = model.state_dict()
        return {
            "state_dict": state_dict,
            "state_dict_sha256": tensor_state_sha256(state_dict),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "t_range": tuple(args.t_range),
            "input_dim": args.input_dim,
            "latent_dim": args.latent_dim,
            "topk_k": args.topk_k,
            "aux_lambda": args.aux_lambda,
            "dead_tolerance": args.dead_tolerance,
            "normalize_decoder": args.normalize_decoder,
            "epoch": epoch,
            "global_step": global_step,
            "training_config": config,
            "validation_metrics": metrics,
            "runtime": runtime_metadata(repo_root),
        }

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "auxiliary": 0.0}
        sample_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, reconstruction_loss, auxiliary_loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if args.normalize_decoder:
                model.normalize_decoder_weights()
            batch_size = batch.shape[0]
            totals["loss"] += loss.item() * batch_size
            totals["reconstruction"] += reconstruction_loss.item() * batch_size
            totals["auxiliary"] += auxiliary_loss.item() * batch_size
            sample_count += batch_size

        averages = {f"train_{key}": value / sample_count for key, value in totals.items()}
        print(
            f"epoch {epoch:03d}/{args.epochs}: loss={averages['train_loss']:.6f} "
            f"recon={averages['train_reconstruction']:.6f} "
            f"aux={averages['train_auxiliary']:.6f}"
        )
        if args.resample_interval > 0 and epoch % args.resample_interval == 0:
            sample = next(iter(loader))[0].to(device)
            print(f"resampled {model.resample_dead_neurons(sample)} dead features")
        metrics = {"epoch": epoch, "global_step": global_step, **averages}
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            valid_metrics = model.compute_eval_metrics(
                valid_loader, device, max_batches=len(valid_loader)
            )
            metrics.update({f"valid_{key}": value for key, value in valid_metrics.items()})
            if valid_metrics["recon_mse"] < best_valid_recon:
                best_valid_recon = float(valid_metrics["recon_mse"])
                best_epoch = epoch
                torch.save(
                    make_checkpoint(epoch, valid_metrics), args.output_dir / "best.pt"
                )
        training_log.append(metrics)

    final_metrics = model.compute_eval_metrics(
        valid_loader, device, max_batches=len(valid_loader)
    )
    torch.save(make_checkpoint(args.epochs, final_metrics), args.output_dir / "final.pt")
    report = {
        "best_epoch": best_epoch,
        "best_valid_recon_mse": best_valid_recon,
        "final_validation": final_metrics,
        "training_curve": training_log,
        "checkpoints": {
            "best": {
                "path": str((args.output_dir / "best.pt").resolve()),
                "sha256": sha256_file(args.output_dir / "best.pt"),
            },
            "final": {
                "path": str((args.output_dir / "final.pt").resolve()),
                "sha256": sha256_file(args.output_dir / "final.pt"),
            },
        },
    }
    write_json(args.output_dir / "metrics.json", report)


if __name__ == "__main__":
    main()