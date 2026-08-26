"""Lightweight contract tests for the DiffLens-style multiplicative baseline."""

from __future__ import annotations

import torch

from Baseline.ig_attribution import aggregate_scores, compute_ig_scores, locate_topk
from Baseline.multiplicative_steer import MultiplicativeSteeringWrapper, parse_group_key
from contsg.models.sae_module import TopKSparseAutoencoder
from sae.shape_classifier import SAEClassClassifier


def test_parse_group_key() -> None:
    assert parse_group_key("beginning/single peak") == (0, 1)
    assert parse_group_key("end/double peaks") == (2, 2)


def test_locate_topk_split() -> None:
    scores = torch.tensor([-0.5, 1.0, -1.2, 0.7, 0.3, -0.1, 2.0, -0.9])
    boost, suppress = locate_topk(scores.numpy(), 3)
    assert boost == [6, 1, 3]
    assert suppress == [2, 7, 0]
    assert set(boost) & set(suppress) == set()


def test_multiplicative_wrapper_edits_only_target_samples() -> None:
    torch.manual_seed(0)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    edits = {"beginning/single peak": {"boost": [1, 2], "suppress": [5]}}
    wrapper = MultiplicativeSteeringWrapper(sae, (0, 3), edits, 2.0, 0.5)
    hidden = torch.randn(6, 4)
    steps = torch.tensor([1, 2])
    wrapper.set_targets(torch.tensor([[1, 0, 0], [0, 0, 0]]))
    transformed, latents = wrapper(hidden, steps)
    assert transformed.shape == hidden.shape
    assert latents.shape == (6, 8)

    z0 = sae.encode(hidden[:3])
    z1 = sae.encode(hidden[3:])
    assert torch.allclose(latents[:3, 1], z0[:, 1] * 2.0)
    assert torch.allclose(latents[:3, 2], z0[:, 2] * 2.0)
    assert torch.allclose(latents[:3, 5], z0[:, 5] * 0.5)
    assert torch.allclose(latents[:3, [0, 3, 4, 6, 7]], z0[:, [0, 3, 4, 6, 7]])
    assert torch.allclose(latents[3:], z1)
    assert torch.allclose(transformed[:3], sae.decode(latents[:3]))
    audit = wrapper.audit()
    assert audit["steered_samples"] == 1
    assert audit["steered_rows"] == 3


def test_multiplicative_wrapper_ignores_out_of_window_steps() -> None:
    torch.manual_seed(0)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    edits = {"beginning/single peak": {"boost": [1], "suppress": [5]}}
    wrapper = MultiplicativeSteeringWrapper(sae, (0, 3), edits)
    hidden = torch.randn(6, 4)
    steps = torch.tensor([4, 4])  # outside the (0, 3) window
    wrapper.set_targets(torch.tensor([[1, 0, 0], [0, 0, 0]]))
    transformed, latents = wrapper(hidden, steps)
    assert torch.allclose(transformed, hidden)
    assert torch.allclose(latents, torch.zeros_like(latents))


def test_ig_completeness_on_tiny_classifier() -> None:
    torch.manual_seed(1)
    classifier = SAEClassClassifier(latent_dim=8, tokens_per_sample=3, hidden_dim=16)
    classifier.eval()
    z = torch.rand(4, 3, 8) * 0.8
    ig = compute_ig_scores(
        classifier, z, stage=0, shape=1, steps=48,
        device=torch.device("cpu"), batch_size=4,
    )
    scores = aggregate_scores(ig)
    assert scores.shape == (8,)

    logp_z = classifier(z).log_softmax(-1)[:, 0, 1].detach()
    logp_0 = classifier(torch.zeros_like(z)).log_softmax(-1)[:, 0, 1].detach()
    error = (ig.sum(dim=-1).mean(dim=1) - (logp_z - logp_0)).abs().max().item()
    assert error < 0.15


def test_multi_hit_samples_merge_ratios_multiplicatively() -> None:
    torch.manual_seed(0)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    edits = {
        "beginning/single peak": {"boost": [1], "suppress": [5]},
        "end/double peaks": {"boost": [5], "suppress": [1]},
        "middle/sag": {"boost": [5]},
    }
    wrapper = MultiplicativeSteeringWrapper(sae, (0, 3), edits, 2.0, 0.5)
    hidden = torch.randn(9, 4)
    steps = torch.tensor([1, 1, 1])
    # s0 hits groups A+B (opposing roles on dims 1 and 5), s1 hits B,
    # s2 hits C+B (same-direction boost on dim 5 stacks, then clamps).
    wrapper.set_targets(torch.tensor([[1, 0, 2], [0, 0, 2], [0, 3, 2]]))
    transformed, latents = wrapper(hidden, steps)

    z = sae.encode(hidden.view(3, 3, 4)).detach()
    # s0: 2.0 * 0.5 == 1.0 on both dims -> net identity edit.
    assert torch.allclose(latents[0:3], z[0])
    # s1: dim1 suppressed, dim5 boosted.
    assert torch.allclose(latents[3:6, 1], z[1, :, 1] * 0.5)
    assert torch.allclose(latents[3:6, 5], z[1, :, 5] * 2.0)
    # s2: dim5 stacked 2.0 * 2.0 = 4.0 -> default clamp caps at 2.0.
    assert torch.allclose(latents[6:9, 1], z[2, :, 1] * 0.5)
    assert torch.allclose(latents[6:9, 5], z[2, :, 5] * 2.0)
    assert torch.allclose(transformed, sae.decode(latents))

    audit = wrapper.audit()
    assert audit["multi_hit_samples"] == 2
    assert audit["steered_samples"] == 3
    assert audit["steered_rows"] == 9


def test_multi_hit_stacking_with_explicit_clamp() -> None:
    torch.manual_seed(0)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    edits = {
        "beginning/single peak": {"boost": [2]},
        "end/double peaks": {"boost": [2]},
    }
    wrapper = MultiplicativeSteeringWrapper(
        sae, (0, 3), edits, 2.0, 0.5, max_ratio=8.0, min_ratio=0.25
    )
    hidden = torch.randn(3, 4)
    wrapper.set_targets(torch.tensor([[1, 0, 2]]))
    _, latents = wrapper(hidden, torch.tensor([1]))
    z = sae.encode(hidden.view(1, 3, 4)).detach()
    assert torch.allclose(latents[:, 2], z[0, :, 2] * 4.0)
    assert wrapper.audit()["multi_hit_samples"] == 1
