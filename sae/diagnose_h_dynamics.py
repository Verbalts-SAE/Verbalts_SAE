"""Control experiment: is gradient dynamics present in the RAW h space?

If the per-state gradient selection sets are just as dynamic in the SAE's
INPUT space (64-dim residual activations), the dynamics cannot be blamed on
the SAE training.  Trains a small MLP classifier on raw h and runs the same
dynamic-vs-fixed analysis as diagnose_gradient_dynamics.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae.diagnose_gradient_dynamics import analyze, per_state_gradients
from sae.latent_classifier import load_activation_cache

D_MODEL = 64  # residual dim at layer 1


class HClassifier(torch.nn.Module):
    """MLP on raw h rows: Linear(64->128) + GELU + Linear(128->12)."""

    def __init__(self) -> None:
        super().__init__()
        self.input_mean = torch.nn.Parameter(torch.zeros(D_MODEL), requires_grad=False)
        self.input_std = torch.nn.Parameter(torch.ones(D_MODEL), requires_grad=False)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_MODEL, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 12),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        standardized = (rows.float() - self.input_mean) / self.input_std.clamp_min(1e-6)
        return self.net(standardized).reshape(-1, 3, 4)


def train_classifier(
    train_rows: torch.Tensor,
    train_labels: torch.Tensor,
    valid_rows: torch.Tensor,
    valid_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
) -> tuple[HClassifier, dict[str, float]]:
    torch.manual_seed(42)
    model = HClassifier().to(device)
    with torch.no_grad():
        model.input_mean.copy_(train_rows.float().mean(dim=0))
        model.input_std.copy_(train_rows.float().std(dim=0).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(train_rows, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = sum(
                criterion(logits[:, stage], labels[:, stage]) for stage in range(3)
            ) / 3
            loss.backward()
            optimizer.step()
            total += float(loss) * len(features)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"h-clf epoch {epoch:03d}: loss={total / len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        logits = model(valid_rows.to(device))
        matches = (logits.argmax(-1) == valid_labels.to(device)).float()
        stage_accuracy = matches.mean(dim=0).cpu().numpy()
    metrics = {
        "mean_stage_accuracy": float(stage_accuracy.mean()),
        "stage_accuracy": stage_accuracy.tolist(),
    }
    print("h-clf valid:", json.dumps(metrics))
    return model, metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path,
                        default=root / "results/sae_retrain/activations_linig/train_layer1_t5_45.pt")
    parser.add_argument("--valid-cache", type=Path,
                        default=root / "results/sae_retrain/activations_linig/valid_layer1_t5_45.pt")
    parser.add_argument("--train-captions", type=Path,
                        default=root / "results/sae_retrain/activations_linig/train_caps_1024.npy")
    parser.add_argument("--valid-captions", type=Path,
                        default=root / "results/sae_retrain/activations_linig/valid_caps_512.npy")
    parser.add_argument("--output-json", type=Path,
                        default=root / "results/h_space_dynamics.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    train_rows, _, train_labels, _ = load_activation_cache(
        args.train_cache, args.train_captions)
    valid_rows, _, valid_labels, _ = load_activation_cache(
        args.valid_cache, args.valid_captions)
    tokens = train_rows.shape[1]
    train_rows = train_rows.reshape(-1, train_rows.shape[-1])
    valid_rows = valid_rows.reshape(-1, valid_rows.shape[-1])
    train_labels = train_labels.repeat_interleave(tokens, dim=0)
    valid_labels = valid_labels.repeat_interleave(tokens, dim=0)
    print(f"h rows: train {tuple(train_rows.shape)} valid {tuple(valid_rows.shape)}")
    model, metrics = train_classifier(
        train_rows, train_labels, valid_rows, valid_labels, device)
    targets = valid_labels[:, 1].to(device)
    grads, probs = per_state_gradients(model, valid_rows.to(device), targets)
    flat_grads = grads.reshape(-1, grads.shape[-1])
    flat_probs = probs
    win = model.net[0].weight.detach()  # [128, 64]
    wout = model.net[2].weight.detach()  # [12, 128]
    fixed = (wout @ win)  # [12, 64] standardized-space composite weights
    fixed = fixed / fixed.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fixed_global = fixed.mean(dim=0)
    fixed_global = fixed_global / fixed_global.norm().clamp_min(1e-9)
    report = {
        "classifier": metrics,
        "mean_target_probability": float(flat_probs.mean()),
        "dynamics": analyze("h-space", flat_grads, flat_probs, fixed_global),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report["dynamics"][k] for k in (
        "pair_jaccard_mean", "fixed_vs_state_jaccard_mean",
        "sign_flip_fraction_vs_fixed", "spearman(state_rank, fixed_rank)_mean")}, indent=1))
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
