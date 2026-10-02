"""Collect residual-layer-1 VerbalTS activations with complete provenance."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import contsg.models.verbalts  # noqa: F401
from contsg.config.schema import ExperimentConfig
from contsg.data.datamodule import TimeSeriesDataset
from contsg.registry import Registry
from sae.provenance import runtime_metadata, sha256_file, tensor_state_sha256, write_json


def prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def parse_target_ranges(value: str) -> list[tuple[int, int]]:
    ranges = []
    for part in value.split(","):
        bounds = [int(item.strip()) for item in part.split("-")]
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise argparse.ArgumentTypeError(f"invalid timestep range: {part!r}")
        ranges.append((bounds[0], bounds[1]))
    return ranges


def load_verbalts(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    config_dict = copy.deepcopy(hparams.get("config", {}))
    if not config_dict:
        raise ValueError(f"checkpoint has no embedded hyper_parameters.config: {checkpoint_path}")
    for key in ("diffusion_embedding_dim", "diffusion_steps"):
        config_dict.pop(key, None)
    if isinstance(config_dict.get("model"), dict):
        for key in ("diffusion_embedding_dim", "diffusion_steps"):
            config_dict["model"].pop(key, None)
    config = ExperimentConfig(**config_dict)
    config.device = str(device)
    config.__dict__["diffusion_steps"] = 50
    config.__dict__["diffusion_embedding_dim"] = 128
    config.model.__dict__["diffusion_steps"] = 50
    config.model.__dict__["diffusion_embedding_dim"] = 128
    model = Registry.get_model("verbalts")(
        config=config,
        learning_rate=hparams.get("learning_rate", 1e-3),
        use_condition=hparams.get("use_condition", True),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    state_hash = tensor_state_sha256(model.state_dict())
    return model.to(device).eval(), state_hash


def load_generator(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, str]:
    """Load any registered generator (verbalts / diffusets / bridge).

    Returns ``(model, model_name)``.  The VerbalTS-specific config fixups from
    ``load_verbalts`` are kept for that model; the other generators rebuild
    directly from the embedded config.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    config_dict = copy.deepcopy(hparams.get("config", {}))
    if not config_dict:
        raise ValueError(f"checkpoint has no embedded hyper_parameters.config: {checkpoint_path}")
    model_cfg = config_dict.get("model", {})
    model_name = model_cfg.get("name") if isinstance(model_cfg, dict) else None
    if not model_name:
        raise ValueError(f"checkpoint config has no model.name: {checkpoint_path}")
    if model_name == "verbalts":
        model, _ = load_verbalts(checkpoint_path, device)
        return model, model_name
    config = ExperimentConfig(**config_dict)
    config.device = str(device)
    model = Registry.get_model(model_name)(
        config=config,
        learning_rate=hparams.get("learning_rate", 1e-3),
        use_condition=hparams.get("use_condition", True),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), model_name


def collect_window(
    model: torch.nn.Module,
    loader: DataLoader,
    t_range: tuple[int, int],
    layer_index: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if layer_index != 1:
        raise ValueError("the approved experiment requires residual layer index 1")
    if layer_index >= len(model.verbalts.residual_layers):
        raise ValueError(f"model has only {len(model.verbalts.residual_layers)} residual layers")
    activations: list[torch.Tensor] = []
    timestep_rows: list[torch.Tensor] = []
    hook_calls = 0
    tokens_per_sample: int | None = None
    current_t: torch.Tensor | None = None

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal hook_calls, tokens_per_sample
        if current_t is None:
            raise RuntimeError("activation hook fired without a current timestep batch")
        residual = output[0] if isinstance(output, (tuple, list)) else output
        if residual.ndim != 4:
            raise ValueError(f"expected residual output (B,C,V,T), got {tuple(residual.shape)}")
        batch_size, channels, n_variables, n_tokens = residual.shape
        rows = residual.permute(0, 2, 3, 1).reshape(-1, channels).detach().cpu().float()
        per_sample = n_variables * n_tokens
        if tokens_per_sample is not None and tokens_per_sample != per_sample:
            raise RuntimeError("residual token count changed during collection")
        tokens_per_sample = per_sample
        activations.append(rows)
        timestep_rows.append(current_t.detach().cpu().repeat_interleave(per_sample))
        hook_calls += 1

    handle = model.verbalts.residual_layers[layer_index].register_forward_hook(hook)
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"collect t={t_range[0]}-{t_range[1]}"):
                batch_data = prepare_batch(batch, device)
                batch_size = int(batch_data["ts"].shape[0])
                current_t = torch.randint(
                    t_range[0], t_range[1] + 1, (batch_size,), generator=generator, device=device
                )
                batch_data["t"] = current_t
                _ = model(batch_data)
    finally:
        handle.remove()
    if not activations or hook_calls != len(loader):
        raise RuntimeError(f"hook audit failed: calls={hook_calls}, expected={len(loader)}")
    activation_tensor = torch.cat(activations)
    timestep_tensor = torch.cat(timestep_rows)
    unique_t, counts = torch.unique(timestep_tensor, return_counts=True)
    audit = {
        "hook_calls": hook_calls,
        "expected_hook_calls": len(loader),
        "tokens_per_sample": tokens_per_sample,
        "activation_rows": int(activation_tensor.shape[0]),
        "feature_dim": int(activation_tensor.shape[1]),
        "timestep_row_counts": {
            str(int(step)): int(count) for step, count in zip(unique_t, counts)
        },
    }
    return activation_tensor, timestep_tensor, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/synth-u"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument(
        "--caption-variant", default="base",
        help="Raw-caption and precomputed-embedding variant used by the checkpoint",
    )
    parser.add_argument("--target-layer-idx", type=int, default=1)
    parser.add_argument("--target-t-ranges", default="40-45,0-45")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional deterministic prefix for smoke tests; omit for complete collection",
    )
    parser.add_argument(
        "--load-bg-attrs",
        action="store_true",
        help="Load {split}_<bg-attrs-file>.npy numeric background attributes into the "
        "batch (required for attribute-conditioned checkpoints)",
    )
    parser.add_argument(
        "--bg-attrs-file",
        default="bg_attrs",
        help="Base name of the {split}_<name>.npy background attribute file "
        "used with --load-bg-attrs (e.g. 'bg_attrs' or 'bg_attrs_v2')",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.ckpt_path.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
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
    model, model_state_hash = load_verbalts(checkpoint_path, device)
    full_dataset = TimeSeriesDataset(
        data_root,
        split=args.split,
        normalize=False,
        caption_variant=args.caption_variant,
        load_bg_attrs=args.load_bg_attrs,
        bg_attrs_filename=args.bg_attrs_file,
    )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        dataset = Subset(full_dataset, range(min(args.max_samples, len(full_dataset))))
    else:
        dataset = full_dataset
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_state_sha256": model_state_hash,
        "data_root": str(data_root),
        "split": args.split,
        "caption_variant": args.caption_variant,
        "load_bg_attrs": bool(args.load_bg_attrs),
        "dataset_size": len(dataset),
        "full_split_size": len(full_dataset),
        "target_layer": args.target_layer_idx,
        "seed": args.seed,
        "runtime": runtime_metadata(repo_root),
    }
    manifest: dict[str, Any] = {"source": source, "caches": []}
    for range_index, t_range in enumerate(parse_target_ranges(args.target_t_ranges)):
        activations, t_values, audit = collect_window(
            model, loader, t_range, args.target_layer_idx, device, args.seed + range_index
        )
        cache_path = output_dir / f"{args.split}_layer1_t{t_range[0]}_{t_range[1]}.pt"
        torch.save(
            {"activations": activations, "t_values": t_values, "t_range": list(t_range),
             "feature_dim": int(activations.shape[1]), "source": source, "audit": audit},
            cache_path,
        )
        entry = {"path": str(cache_path), "sha256": sha256_file(cache_path), **audit}
        manifest["caches"].append(entry)
        print(f"saved {cache_path}: {tuple(activations.shape)}, sha256={entry['sha256']}")
    write_json(output_dir / f"{args.split}_manifest.json", manifest)


if __name__ == "__main__":
    main()