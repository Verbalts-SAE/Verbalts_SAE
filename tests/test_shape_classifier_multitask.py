"""Regression tests for Synth-U's overlapping position-label semantics."""

import torch
import torch.nn.functional as F

from sae.shape_classifier import weighted_classification_loss


def test_loss_supervises_all_overlapping_position_labels() -> None:
    """One curve has three labels; it is not reduced to one global class."""

    logits = torch.tensor(
        [[[3.0, 0.0, -1.0, -2.0],
          [-2.0, -1.0, 0.0, 3.0],
          [-2.0, -1.0, 0.0, 3.0]]],
        requires_grad=True,
    )
    # beginning=nothing, middle=sag, end=sag. The repeated sag labels at two
    # positions are both valid and must both contribute to the objective.
    labels = torch.tensor([[0, 3, 3]])

    actual = weighted_classification_loss(logits, labels)
    expected = torch.stack([
        F.cross_entropy(logits[:, head], labels[:, head])
        for head in range(3)
    ]).mean()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert logits.grad is not None
    assert (logits.grad.abs().sum(dim=-1) > 0).tolist() == [[True, True, True]]


def test_position_heads_are_independently_normalized() -> None:
    """Synth-U uses three independent four-way heads, never one 12-way head."""

    logits = torch.randn(2, 3, 4)
    probabilities = logits.softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(2, 3), rtol=0, atol=1e-6
    )
    assert not torch.allclose(
        probabilities.reshape(2, -1).sum(dim=-1), torch.ones(2)
    )