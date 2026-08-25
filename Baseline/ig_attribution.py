"""Integrated-Gradients (IG) feature location on SAE latents (DiffLens style).

The DiffLens pipeline locates the latent dimensions whose activity drives a
target class probability, then edits them multiplicatively at inference
time.  Here the attribution target is the log-probability of one
(stage, shape) head of the frozen position-aware latent classifier, and the
attributed inputs are the per-token Top-K SAE latents ``z``.

Because the SAE latents are sparse and non-negative, the natural IG baseline
is the all-zero latent vector: every attribution path runs from ``0`` to the
observed ``z``, so ``IG_i = z_i * mean_k grad_i(z * k / steps)``.  Positive
IG dimensions support the target class (candidates for boosting); negative
IG dimensions oppose it (candidates for suppressing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sae.shape_classifier import SAEClassClassifier


def compute_ig_scores(
    classifier: SAEClassClassifier,
    latents: torch.Tensor,
    stage: int,
    shape: int,
    steps: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return per-token IG scores with shape ``(samples, tokens, latent_dim)``.

    ``classifier`` is put into eval mode (dropout disabled) before the
    attribution pass.  The returned tensor lives on CPU in float32; the
    per-dimension aggregate score is computed by :func:`aggregate_scores`.
    """

    if latents.ndim != 3:
        raise ValueError(
            f"latents must have shape (samples, tokens, D), got {tuple(latents.shape)}"
        )
    if latents.shape[1] != classifier.tokens_per_sample:
        raise ValueError(
            f"latent token count {latents.shape[1]} does not match classifier "
            f"tokens_per_sample {classifier.tokens_per_sample}"
        )
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if not 0 <= stage < 3 or not 0 <= shape < 4:
        raise ValueError(f"invalid (stage, shape) = ({stage}, {shape})")

    classifier.eval()
    output = torch.empty(latents.shape, dtype=torch.float32)
    for start in range(0, latents.shape[0], batch_size):
        chunk = latents[start:start + batch_size].detach().float().to(device)
        accumulated = torch.zeros_like(chunk)
        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = (chunk * alpha).requires_grad_(True)
            logits = classifier(interpolated)  # [B, 3, 4]
            log_probability = logits.log_softmax(dim=-1)[:, stage, shape].sum()
            (gradient,) = torch.autograd.grad(log_probability, interpolated)
            accumulated += gradient.detach()
        ig = (chunk * (accumulated / steps)).cpu()
        output[start:start + len(chunk)] = ig
        del chunk, accumulated, ig
    return output


def aggregate_scores(ig: torch.Tensor) -> np.ndarray:
    """Mean per-dimension IG score over all samples and token positions."""

    if ig.ndim != 3:
        raise ValueError(f"expected (samples, tokens, D), got {tuple(ig.shape)}")
    scores = ig.float().mean(dim=(0, 1)).cpu().numpy()
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def locate_topk(scores: np.ndarray, topk: int) -> tuple[list[int], list[int]]:
    """Split per-dimension scores into boost and suppress latent dims.

    ``boost`` holds the ``topk`` dimensions with the largest (most positive)
    scores; ``suppress`` holds the ``topk`` dimensions with the smallest
    (most negative) scores.  The two sets never overlap.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if topk < 1 or topk > len(scores):
        raise ValueError(f"topk {topk} is invalid for {len(scores)} dimensions")
    order = np.argsort(scores)
    boost = [int(dim) for dim in order[::-1][:topk]]
    suppress = [int(dim) for dim in order[:topk]]
    if set(boost) & set(suppress):
        raise ValueError("boost and suppress dimension sets overlap")
    return boost, suppress


def load_located_features(path: str | Path) -> dict[str, Any]:
    """Load the located-features JSON artifact written by ``locate_features``."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"located-features artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"target", "boost_dims", "suppress_dims", "topk"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"located-features artifact is missing keys: {sorted(missing)}")
    return payload


__all__ = [
    "aggregate_scores",
    "compute_ig_scores",
    "load_located_features",
    "locate_topk",
]
