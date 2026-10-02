"""Diagnose latent steering geometry from activation caches.

Walk real-waveform SAE latents along the MLP gradient toward the cyclic
target class and measure argmax/logits response.  No generator needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae.shape_classifier import load_classifier_checkpoint
from sae.steering import load_sae_checkpoint


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-checkpoint", type=Path, required=True)
    p.add_argument("--latent-classifier", type=Path, required=True)
    p.add_argument("--activation-cache", type=Path, required=True)
    p.add_argument("--attrs", type=Path, required=True,
                   help="attrs_idx.npy matching the cache samples")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--steps", default="0.0,0.1,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sae, t_range, _ = load_sae_checkpoint(args.sae_checkpoint.resolve(), device)
    mlp, payload = load_classifier_checkpoint(args.latent_classifier.resolve(), device)
    heads = int(payload.get("num_heads", 3))
    cache = torch.load(args.activation_cache.resolve(), map_location="cpu")
    hidden = cache["activations"].float()
    t_values = cache["t_values"].numpy().astype(np.int64)
    attrs = np.load(args.attrs.resolve()).astype(np.int64)
    if attrs.ndim == 1:
        attrs = attrs.reshape(-1, 1)
    rows_per_sample = len(hidden) // len(attrs)
    limit = min(args.max_samples, len(attrs))
    hidden = hidden[:limit * rows_per_sample]
    attrs = attrs[:limit]
    print(f"cache rows={len(hidden)} samples={limit} heads={heads} "
          f"t_range={t_range} rows_per_sample={rows_per_sample}")

    steps = [float(x) for x in args.steps.split(",") if x.strip()]
    with torch.no_grad():
        latents = torch.cat([
            sae(hidden[s:s + args.batch_size].to(device))[1].detach().cpu()
            for s in range(0, len(hidden), args.batch_size)
        ])
    # restore sample/token boundaries: (samples, tokens, latent_dim)
    z = latents.view(limit, rows_per_sample, -1)
    per_head_report = {}
    for head in range(heads):
        y = torch.from_numpy(attrs[:, head]).to(device)
        target = (y + 1) % 4
        z_g = z.to(device).clone().requires_grad_(True)
        logits = mlp(z_g)[:, head]
        loss = torch.nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, z_g)[0]
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        grad_dir = grad / grad_norm
        baseline = logits.argmax(-1)
        rows = {}
        for step in steps:
            with torch.no_grad():
                moved_logits = mlp(z_g + step * grad_dir)[:, head]
            pred = moved_logits.argmax(-1)
            tgt_prob = torch.softmax(moved_logits, -1).gather(1, target[:, None]).squeeze(-1)
            rows[str(step)] = {
                "target_hit": float((pred == target).float().mean()),
                "orig_kept": float((pred == baseline).float().mean()),
                "mean_target_prob": float(tgt_prob.mean()),
                "mean_grad_norm": float(grad_norm.mean()),
            }
        per_head_report[f"head{head}"] = {
            "baseline_target_hit": float((baseline == target).float().mean()),
            "per_class": {
                str(c): {
                    "n": int((y == c).sum()),
                    "baseline_target_hit":
                        float((baseline[y == c] == target[y == c]).float().mean()),
                    "target_was": int((c + 1) % 4),
                } for c in range(4)
            },
            "walk": rows,
        }
    report = {"samples": limit, "heads": heads, "t_range": list(t_range),
              "steps": steps, "per_head": per_head_report}
    (output / "geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({h: {s: f"{v['target_hit']:.3f}" for s, v in r["walk"].items()}
                      for h, r in per_head_report.items()}, indent=2))


if __name__ == "__main__":
    main()
