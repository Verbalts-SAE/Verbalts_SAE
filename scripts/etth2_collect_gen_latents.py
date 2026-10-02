"""Collect generated-domain latents for on-policy MLP training.

Generates pure (unsteered) curves for a split prefix and records the layer-1
residual at every guidance-range step t in [5, 45], producing:

- ``curves.npy``                (N, 128) generated final curves
- ``gen_latents_cache.pt``      activation cache in train_classifier format
- ``labels_captions.npy``       per-row caption strings (labels repeated per step)
- ``labels_shape_ids.npy``      (rows, 3) integer segment labels (labels repeated)

The cache is loadable by ``sae.shape_classifier.load_activation_cache`` so the
same ``sae.train_classifier`` entry point can train an on-policy MLP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import contsg.models.verbalts  # noqa: F401
from sae.activations import load_verbalts
from sae.provenance import write_json
from sae.shape_classifier import captions_to_shape_ids

T_MIN, T_MAX = 5, 45


class CollectWrapper(torch.nn.Module):
    """Pass-through transform recording the flattened residual per timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.hiddens: list[torch.Tensor] = []
        self.steps: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor, diffusion_step: torch.Tensor):
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        self.hiddens.append(hidden.detach().cpu().float())
        self.steps.append(steps.cpu())
        return hidden, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--n-samples", type=int, default=2048,
                        help="0 collects the whole split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = args.data_root
    captions = np.load(data / f"{args.split}_text_caps.npy", allow_pickle=True).reshape(-1)
    embeddings = np.load(data / f"{args.split}_cap_emb.npy").astype(np.float32)
    total = min(len(captions), args.n_samples if args.n_samples > 0 else len(captions))
    captions = captions[:total]
    embeddings = embeddings[:total]

    model, _ = load_verbalts(args.model_checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapper = CollectWrapper()
    model.verbalts.activation_transform = wrapper
    curves_parts = []
    try:
        with torch.no_grad():
            for start in range(0, total, args.batch_size):
                condition = torch.from_numpy(embeddings[start:start + args.batch_size]).to(device)
                torch.manual_seed(args.seed + start)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + start)
                generated = model.generate(condition, n_samples=1, sampler="ddim")[0, :, :, 0]
                curves_parts.append(generated.detach().cpu().float().numpy())
    finally:
        model.verbalts.activation_transform = None
    curves = np.concatenate(curves_parts, axis=0)
    np.save(out / "curves.npy", curves)

    # Reorder collected activations into (row, C) aligned with (sample, step).
    # Chunks are appended batch-major inside per_step, so per_step[t][k] is sample k.
    tokens = None
    per_step: dict[int, list[torch.Tensor]] = {}
    for hidden, steps in zip(wrapper.hiddens, wrapper.steps):
        per_sample_rows = hidden.shape[0] // steps.numel()
        for row, step in enumerate(steps.tolist()):
            chunk = hidden[row * per_sample_rows:(row + 1) * per_sample_rows]
            if tokens is None:
                tokens = chunk.shape[0]
            per_step.setdefault(step, []).append(chunk)
    if tokens is None:
        raise ValueError("no activations collected")
    rows_list: list[torch.Tensor] = []
    t_list: list[int] = []
    shape_ids_list: list[np.ndarray] = []
    captions_list: list[str] = []
    shape_ids = captions_to_shape_ids(captions).numpy()
    for step in sorted(per_step):
        if not T_MIN <= step <= T_MAX:
            continue
        for chunk_index, chunk in enumerate(per_step[step]):
            b = chunk.shape[0]
            rows_list.append(chunk)
            t_list.extend([step] * b)
            shape_ids_list.append(shape_ids[chunk_index])
            captions_list.append(str(captions[chunk_index]))
    activations = torch.cat(rows_list, dim=0)
    t_values = torch.tensor(t_list, dtype=torch.long)
    labels_shape_ids = np.stack(shape_ids_list)
    labels_captions = np.asarray(captions_list, dtype=object)
    torch.save(
        {"activations": activations, "t_values": t_values, "t_range": [T_MIN, T_MAX],
         "audit": {"tokens_per_sample": tokens}, "feature_dim": int(activations.shape[1])},
        out / "gen_latents_cache.pt",
    )
    np.save(out / "labels_shape_ids.npy", labels_shape_ids)
    np.save(out / "labels_captions.npy", labels_captions)
    write_json(out / "collection_manifest.json", {
        "split": args.split, "samples": total,
        "steps_per_sample": len([s for s in per_step if T_MIN <= s <= T_MAX]),
        "rows": int(activations.shape[0]), "tokens_per_sample": tokens,
        "seed": args.seed, "n_generated_curves": int(curves.shape[0]),
    })
    print(json.dumps({"split": args.split, "samples": total, "rows": int(activations.shape[0]),
                      "tokens_per_sample": tokens}, indent=1))
    print("GEN_LATENT_COLLECTION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
