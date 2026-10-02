"""Diagnose the MLP domain gap on etth2_2class: GT-latent vs generated-latent.

Task A (MLP-on-GT): encode GT test curves at a random diffusion step in
[5, 45] through the VerbalTS layer-1 residual, SAE-encode, classify with the
trained SAEClassClassifier, and compare against caption segment labels.

Task B (MLP-on-generated): run the pure DDIM sampler, collecting the layer-1
residual at every guidance-range step t in [5, 45], SAE-encode each, classify
with the MLP, and compare against the same ground-truth labels (reporting both
the per-step ACCR curve and a per-sample majority vote across steps).

The gap between the two ACCR values bounds what steering can hope to achieve:
steering follows the MLP gradient on generated latents, so if Task B ACCR is
barely above the CNN ACCR of pure generations, the gradient is blind on the
generated domain rather than the intervention being too weak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import contsg.models.verbalts  # noqa: F401
from contsg.data.datamodule import TimeSeriesDataset
from sae.activations import load_verbalts, prepare_batch
from sae.eval_guidance import generate_variant
from sae.provenance import write_json
from sae.shape_classifier import captions_to_shape_ids, classifier_metrics, encode_sae_latents, load_classifier_checkpoint
from sae.steering import load_sae_checkpoint

T_MIN, T_MAX = 5, 45


class CollectWrapper(torch.nn.Module):
    """Pass-through transform that records the flattened residual per timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.hiddens: list[torch.Tensor] = []
        self.steps: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor, diffusion_step: torch.Tensor):
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        self.hiddens.append(hidden.detach().cpu().float())
        self.steps.append(steps.cpu())
        return hidden, None


@torch.no_grad()
def mlp_accr_on_gt(
    model: torch.nn.Module,
    loader: DataLoader,
    sae,
    mlp,
    labels: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Task A: classify GT curves encoded at random t in [5, 45]."""
    layer = model.verbalts.residual_layers[1]
    activations: list[torch.Tensor] = []
    current_t: torch.Tensor | None = None

    def hook(_m, _i, output) -> None:
        nonlocal current_t
        residual = output[0] if isinstance(output, (tuple, list)) else output
        rows = residual.permute(0, 2, 3, 1).reshape(-1, residual.shape[1]).detach().cpu().float()
        activations.append(rows)

    generator = torch.Generator(device=device).manual_seed(seed)
    handle = layer.register_forward_hook(hook)
    try:
        for batch in loader:
            batch_data = prepare_batch(batch, device)
            size = int(batch_data["ts"].shape[0])
            current_t = torch.randint(T_MIN, T_MAX + 1, (size,), generator=generator, device=device)
            batch_data["t"] = current_t
            _ = model(batch_data)
    finally:
        handle.remove()
    rows = torch.cat(activations)
    n_samples = len(labels)
    if rows.shape[0] % n_samples:
        raise ValueError("activation rows cannot be restored to samples")
    tokens_per_sample = rows.shape[0] // n_samples
    samples = rows.reshape(n_samples, tokens_per_sample, -1)
    latents = encode_sae_latents(samples, sae, device, batch_size=64)
    return classifier_metrics(mlp, latents, labels, device, batch_size=256)


@torch.no_grad()
def mlp_accr_on_generated(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    sae,
    mlp,
    labels: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Task B: classify pure-generation intermediate latents at each t in [5, 45]."""
    wrapper = CollectWrapper()
    model.verbalts.activation_transform = wrapper
    try:
        for start in range(0, len(embeddings), batch_size):
            condition = torch.from_numpy(embeddings[start:start + batch_size]).to(device)
            torch.manual_seed(seed + start)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + start)
            model.generate(condition, n_samples=1, sampler="ddim")[0, :, :, 0]
    finally:
        model.verbalts.activation_transform = None
    # Reorder: hiddens[i] corresponds to the diffusion step steps[i] (B rows).
    per_step: dict[int, list[torch.Tensor]] = {}
    tokens = None
    for hidden, steps in zip(wrapper.hiddens, wrapper.steps):
        per_sample_rows = hidden.shape[0] // steps.numel()
        for row, step in enumerate(steps.tolist()):
            if not T_MIN <= step <= T_MAX:
                continue
            chunk = hidden[row * per_sample_rows:(row + 1) * per_sample_rows]
            if tokens is None:
                tokens = chunk.shape[0]
            per_step.setdefault(step, []).append(chunk)
    if tokens is None:
        raise ValueError("no guidance-range activations collected")
    per_step_accr: dict[str, float] = {}
    per_step_stage: dict[str, float] = {}
    votes = torch.zeros(len(labels), mlp.num_heads, mlp.num_shape_classes, dtype=torch.float32)
    for step in sorted(per_step):
        chunks = per_step[step]
        n_samples = len(chunks)
        samples = torch.empty(n_samples, tokens, sae.input_dim, dtype=torch.float32)
        for i, chunk in enumerate(chunks):
            samples[i] = chunk
        # chunks were appended in generation order, so rows stay aligned with labels
        latents = encode_sae_latents(samples, sae, device, batch_size=64)
        metrics = classifier_metrics(mlp, latents, labels, device, batch_size=256)
        per_step_accr[str(step)] = metrics["accr"]
        per_step_stage[str(step)] = metrics["mean_stage_accuracy"]
        logits = mlp(latents.to(device, dtype=torch.float32))
        probs = logits.softmax(dim=-1).cpu()
        votes += probs
    vote_pred = votes.argmax(dim=-1)
    vote_matches = vote_pred == labels
    return {
        "accr_vote": float(vote_matches.all(dim=1).float().mean()),
        "stage_acc_vote": float(vote_matches.float().mean()),
        "per_step_accr": per_step_accr,
        "per_step_stage": per_step_stage,
        "steps_seen": len(per_step),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    root = args.results_root
    model, _ = load_verbalts(args.model_checkpoint, device)
    sae, _, _ = load_sae_checkpoint(root / "models/t5_45/best.pt", device)
    mlp, _ = load_classifier_checkpoint(root / "classwise_classifiers/t5_45/best.pt", device)
    for network in (model, sae, mlp):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)

    data = args.data_root
    full_dataset = TimeSeriesDataset(data, split="test", normalize=False, caption_variant="base")
    dataset = Subset(full_dataset, range(min(args.n_samples, len(full_dataset))))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    captions = np.load(data / "test_text_caps.npy", allow_pickle=True).reshape(-1)[:len(dataset)]
    labels = captions_to_shape_ids(captions)
    embeddings = np.load(data / "test_cap_emb.npy")[:len(dataset)].astype(np.float32)

    report = {"n_samples": len(dataset), "seed": args.seed}
    report["mlp_on_gt"] = mlp_accr_on_gt(model, loader, sae, mlp, labels, device, args.seed)
    report["mlp_on_generated"] = mlp_accr_on_generated(
        model, embeddings, sae, mlp, labels, device, args.seed)
    write_json(out / "mlp_domain_diag.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "mlp_on_generated"}
                     | {"mlp_on_generated": {k: v for k, v in report["mlp_on_generated"].items()
                                             if k not in ("per_step_accr", "per_step_stage")}},
                     indent=1))
    print("MLP_DOMAIN_DIAG_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
