"""SAE-free control classifier on the raw 64-dim residual stream (h space).

HClassifier maps pooled raw-h rows to the same 3-by-4 segment-shape
distribution as the SAE-latent classifier.  It is used by the ablation
runner as the ``hguidance`` control, proving that the SAE latents carry
steering signal beyond the input dynamics.
"""

from __future__ import annotations

import json

import numpy as np
import torch


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
    """Fit the h-space probe and return it with stage-accuracy metrics."""

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
