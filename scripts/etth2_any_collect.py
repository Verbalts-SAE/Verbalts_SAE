"""Collect GT-forward activations for verbalts/diffusets/bridge on etth2_2class.

Attaches a CollectWrapper to the model's activation-transform hook, runs the
model forward on real batches (random diffusion steps like training), and
saves rows whose step falls in [t_min, t_max] as an SAE training cache plus
sample-boundary-consistent filtered label files for MLP training.

Usage:
  python scripts/etth2_any_collect.py --model-checkpoint CKPT --split train \
      --output-dir DIR --model-type auto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.data.datamodule import TimeSeriesDataset
from sae.activations import load_generator


class CollectWrapper(torch.nn.Module):
    """Collect hidden rows within an inclusive timestep window (identity transform)."""

    def __init__(self, t_min: int, t_max: int) -> None:
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.hiddens: list[torch.Tensor] = []
        self.steps: list[torch.Tensor] = []
        self.kept_samples: list[torch.Tensor] = []
        self.tokens_per_sample: int | None = None
        self.forward_calls = 0

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        if steps.numel() == 0 or hidden.shape[0] % steps.numel() != 0:
            raise ValueError("flattened activation count must be divisible by timestep count")
        tokens = hidden.shape[0] // steps.numel()
        if self.tokens_per_sample is None:
            self.tokens_per_sample = tokens
        elif self.tokens_per_sample != tokens:
            raise RuntimeError("token count changed during collection")
        row_steps = steps.repeat_interleave(tokens)
        selected = (row_steps >= self.t_min) & (row_steps <= self.t_max)
        self.hiddens.append(hidden.detach().float()[selected].cpu())
        self.steps.append(row_steps[selected].cpu())
        self.kept_samples.append(selected.reshape(-1, tokens).any(dim=1).cpu())
        self.forward_calls += 1
        return hidden, hidden


def attach_collector(
    model: torch.nn.Module, model_name: str, wrapper: CollectWrapper,
    transform_block_index: int | None = 1,
) -> None:
    core = {
        "verbalts": getattr(model, "verbalts", model),
        "diffusets": getattr(model, "unet", model),
        "bridge": getattr(model, "model", model),
    }[model_name]
    # DiffuSETS: fire the hook after the selected down block (default: 2nd
    # block, L=4 tokens, C=16) instead of the legacy bottleneck position
    # (L=1, C=64) so that segment position information is preserved for the
    # 3-head MLP.
    if model_name == "diffusets" and hasattr(core, "transform_block_index"):
        core.transform_block_index = transform_block_index
    core.activation_transform = wrapper


def detach_collector(model: torch.nn.Module, model_name: str) -> None:
    core = {
        "verbalts": getattr(model, "verbalts", model),
        "diffusets": getattr(model, "unet", model),
        "bridge": getattr(model, "model", model),
    }[model_name]
    core.activation_transform = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--t-min", type=int, default=5)
    parser.add_argument("--t-max", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--transform-block-index", type=int, default=1,
        help="DiffuSETS down-block index after which the hook fires (0..3; "
             "higher = deeper, fewer tokens per sample)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model, model_name = load_generator(args.model_checkpoint, device)
    print(f"MODEL_TYPE={model_name}", flush=True)

    data = TimeSeriesDataset(
        data_folder=args.data_root,
        split=args.split,
        normalize=False,
        provide_bridge_example=(model_name == "bridge"),
        caption_variant="base",
    )
    dataset = data
    max_samples = args.max_samples or len(dataset)
    indices = np.arange(min(max_samples, len(dataset)))
    torch.manual_seed(args.seed)

    wrapper = CollectWrapper(args.t_min, args.t_max)
    attach_collector(model, model_name, wrapper, args.transform_block_index)
    kept_mask: list[bool] = []
    try:
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(indices), args.batch_size), desc=f"collect {args.split}"):
                idx = indices[batch_start:batch_start + args.batch_size]
                rows = [dataset[int(i)] for i in idx]
                ts = torch.stack([r["ts"] for r in rows]).to(device)  # (B, L, C)
                tp = torch.stack([r["tp"] for r in rows]).to(device)  # (B, L)
                cap_emb = torch.stack([r["cap_emb"] for r in rows]).to(device)  # (B, D)
                batch = {"ts": ts, "tp": tp, "cap_emb": cap_emb}
                if model_name == "bridge":
                    batch["bridge_example_ts"] = ts.clone()
                elif model_name == "verbalts":
                    batch["t"] = torch.randint(args.t_min, args.t_max + 1, (len(idx),), device=device)
                model(batch)
                kept_mask.extend(wrapper.kept_samples[-1].tolist())
    finally:
        detach_collector(model, model_name)

    activations = torch.cat(wrapper.hiddens)
    t_values = torch.cat(wrapper.steps)
    tokens = wrapper.tokens_per_sample or 1
    cache = {
        "activations": activations,
        "t_values": t_values,
        "audit": {
            "tokens_per_sample": tokens,
            "forward_calls": wrapper.forward_calls,
        },
        "t_range": [args.t_min, args.t_max],
    }
    torch.save(cache, out / f"{args.split}_activation_cache.pt")
    kept = np.asarray(kept_mask)
    for stem in ("attrs_idx",):
        labels = np.load(args.data_root / f"{args.split}_{stem}.npy")
        filtered = labels[np.arange(len(indices))[kept]]
        np.save(out / f"{args.split}_filtered_{stem}.npy", filtered)
    manifest = {
        "model_checkpoint": str(args.model_checkpoint),
        "model_type": model_name,
        "split": args.split,
        "activation_rows": int(activations.shape[0]),
        "feature_dim": int(activations.shape[1]),
        "tokens_per_sample": tokens,
        "samples_forwarded": int(len(indices)),
        "samples_kept": int(kept.sum()),
        "t_range": [args.t_min, args.t_max],
    }
    (out / "collection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
