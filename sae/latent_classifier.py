"""Lightweight class classifier operating directly on pooled SAE latents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contsg.models.sae_module import TopKSparseAutoencoder
from sae.visualize_variants import SHAPE_NAMES, STAGE_NAMES, parse_segment_shapes


CLASSIFIER_SCHEMA_VERSION = 1


def captions_to_shape_ids(captions: np.ndarray) -> torch.Tensor:
    """Return caption-derived labels with shape ``(samples, 3)``."""

    shape_to_id = {name: index for index, name in enumerate(SHAPE_NAMES)}
    labels = [
        [shape_to_id[shape] for shape in parse_segment_shapes(str(caption))]
        for caption in captions.reshape(-1)
    ]
    return torch.tensor(labels, dtype=torch.long)


def class_group_key(stage_index: int, shape_index: int) -> str:
    return f"{STAGE_NAMES[stage_index].lower()}/{SHAPE_NAMES[shape_index]}"


def iter_class_groups() -> list[tuple[int, int, str]]:
    return [
        (stage, shape, class_group_key(stage, shape))
        for stage in range(len(STAGE_NAMES))
        for shape in range(len(SHAPE_NAMES))
    ]


class SAEClassClassifier(nn.Module):
    """Predict three four-way shape classes from position-aware SAE latents."""

    def __init__(
        self,
        latent_dim: int,
        tokens_per_sample: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.tokens_per_sample = int(tokens_per_sample)
        self.hidden_dim = int(hidden_dim)
        self.dropout_probability = float(dropout)
        if input_mean is None:
            input_mean = torch.zeros(self.latent_dim)
        if input_std is None:
            input_std = torch.ones(self.latent_dim)
        self.register_buffer("input_mean", input_mean.detach().float().clone())
        self.register_buffer("input_std", input_std.detach().float().clone().clamp_min(1e-6))
        self.token_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(self.tokens_per_sample, self.hidden_dim)
        )
        self.stage_queries = nn.Parameter(torch.empty(len(STAGE_NAMES), self.hidden_dim))
        self.dropout = nn.Dropout(self.dropout_probability)
        self.output = nn.Linear(self.hidden_dim, len(SHAPE_NAMES))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.stage_queries, std=0.02)

    def forward(self, sample_latents: torch.Tensor) -> torch.Tensor:
        if sample_latents.ndim != 3:
            raise ValueError("sample_latents must have shape (samples, tokens, latent_dim)")
        if sample_latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent dimension {self.latent_dim}, got {sample_latents.shape[-1]}"
            )
        if sample_latents.shape[1] != self.tokens_per_sample:
            raise ValueError(
                f"expected {self.tokens_per_sample} tokens, got {sample_latents.shape[1]}"
            )
        standardized = (sample_latents.float() - self.input_mean) / self.input_std
        tokens = self.token_projection(standardized) + self.position_embedding.unsqueeze(0)
        tokens = self.dropout(torch.nn.functional.gelu(tokens))
        attention_logits = torch.einsum(
            "bth,sh->bst", tokens, self.stage_queries
        ) / self.hidden_dim ** 0.5
        attention = attention_logits.softmax(dim=-1)
        stage_features = torch.einsum("bst,bth->bsh", attention, tokens)
        return self.output(self.dropout(stage_features))


def load_activation_cache(
    cache_path: str | Path,
    captions_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load ordered activation rows and restore sample/token boundaries."""

    cache_path = Path(cache_path).resolve()
    captions_path = Path(captions_path).resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"activations", "t_values", "audit", "t_range"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"activation cache is missing keys: {sorted(missing)}")
    activations = payload["activations"].float()
    timesteps = payload["t_values"].long()
    tokens_per_sample = int(payload["audit"]["tokens_per_sample"])
    if activations.ndim != 2 or len(activations) % tokens_per_sample:
        raise ValueError("activation rows cannot be restored to complete samples")
    sample_count = len(activations) // tokens_per_sample
    if timesteps.shape != (len(activations),):
        raise ValueError("timestep rows do not match activation rows")
    sample_steps = timesteps.reshape(sample_count, tokens_per_sample)
    if not torch.all(sample_steps == sample_steps[:, :1]):
        raise ValueError("tokens belonging to one sample have inconsistent timesteps")
    captions = np.load(captions_path, allow_pickle=True)
    labels = captions_to_shape_ids(captions)
    if labels.shape != (sample_count, len(STAGE_NAMES)):
        raise ValueError(
            f"caption labels {tuple(labels.shape)} do not match {sample_count} cached samples"
        )
    samples = activations.reshape(sample_count, tokens_per_sample, activations.shape[-1])
    return samples, sample_steps[:, 0], labels, payload


@torch.no_grad()
def encode_sae_latents(
    activation_samples: torch.Tensor,
    sae: TopKSparseAutoencoder,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode activation samples once, preserving all token positions."""

    if activation_samples.ndim != 3:
        raise ValueError("activation_samples must have shape (samples, tokens, hidden_dim)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sample_count, tokens_per_sample, _ = activation_samples.shape
    encoded = torch.empty(
        sample_count, tokens_per_sample, sae.latent_dim, dtype=torch.float16
    )
    sae.eval()
    for start in range(0, sample_count, batch_size):
        hidden = activation_samples[start:start + batch_size].to(device)
        rows = hidden.reshape(-1, hidden.shape[-1])
        latents = sae.encode(rows).reshape(hidden.shape[0], tokens_per_sample, sae.latent_dim)
        encoded[start:start + len(hidden)] = latents.to(dtype=torch.float16, device="cpu")
    return encoded


def weighted_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    for stage in range(len(STAGE_NAMES)):
        weight = None if class_weights is None else class_weights[stage]
        losses.append(F.cross_entropy(logits[:, stage], labels[:, stage], weight=weight))
    return torch.stack(losses).mean()


@torch.no_grad()
def classifier_metrics(
    model: SAEClassClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    probability_parts = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start:start + batch_size].to(device, dtype=torch.float32))
        predictions.append(logits.argmax(dim=-1).cpu())
        probability_parts.append(logits.softmax(dim=-1).cpu())
    predicted = torch.cat(predictions)
    probabilities = torch.cat(probability_parts)
    matches = predicted == labels
    groups: dict[str, Any] = {}
    for stage, shape, key in iter_class_groups():
        support = labels[:, stage] == shape
        predicted_support = predicted[:, stage] == shape
        true_positive = int((support & predicted_support).sum())
        groups[key] = {
            "support": int(support.sum()),
            "accuracy_on_support": float(matches[support, stage].float().mean()) if support.any() else 0.0,
            "precision": true_positive / max(int(predicted_support.sum()), 1),
            "mean_target_probability_on_support": (
                float(probabilities[support, stage, shape].mean()) if support.any() else 0.0
            ),
        }
    return {
        "joint_three_stage_accuracy": float(matches.all(dim=1).float().mean()),
        "mean_stage_accuracy": float(matches.float().mean()),
        "stage_accuracy": {
            STAGE_NAMES[stage].lower(): float(matches[:, stage].float().mean())
            for stage in range(len(STAGE_NAMES))
        },
        "groups": groups,
        "samples": len(features),
    }


def load_classifier_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SAEClassClassifier, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != CLASSIFIER_SCHEMA_VERSION:
        raise ValueError(f"unsupported latent classifier checkpoint schema: {path}")
    model = SAEClassClassifier(
        latent_dim=int(checkpoint["latent_dim"]),
        tokens_per_sample=int(checkpoint["tokens_per_sample"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        input_mean=checkpoint["input_mean"],
        input_std=checkpoint["input_std"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


__all__: Sequence[str] = (
    "CLASSIFIER_SCHEMA_VERSION",
    "SAEClassClassifier",
    "captions_to_shape_ids",
    "class_group_key",
    "classifier_metrics",
    "iter_class_groups",
    "load_activation_cache",
    "load_classifier_checkpoint",
    "encode_sae_latents",
    "weighted_classification_loss",
)