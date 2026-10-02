import torch
from torch import nn

from contsg.models.sae_module import TopKSparseAutoencoder
from sae.steering import LatentClassifierGuidanceWrapper


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(8, 4)
        self.register_buffer("input_std", torch.ones(8))

    def forward(self, z):
        return self.output(z)


def test_trace_does_not_change_intervention_and_resets():
    torch.manual_seed(9)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    classifier = TinyClassifier()
    wrappers = [LatentClassifierGuidanceWrapper(
        sae, (5, 45), classifier, eta=10, topk=2, topk_mode="dynamic",
        diagnostic_trace=enabled) for enabled in (False, True)]
    hidden = torch.randn(6, 4)
    for wrapper in wrappers:
        wrapper.set_targets(torch.tensor([[0, 1, 2], [1, 2, 3]]))
    with torch.no_grad():
        plain = wrappers[0](hidden, torch.tensor([20, 20]))
        traced = wrappers[1](hidden, torch.tensor([20, 20]))
    for a, b in zip(plain, traced):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(wrappers[1].trace_records) == 2
    for row in wrappers[1].trace_records:
        assert len(row["probability_before"]) == 3
        assert len(row["probability_roundtrip_before"]) == 3
        assert len(row["probability_roundtrip_after"]) == 3
        assert row["hidden_delta_rmse"] >= 0
        assert row["hidden_delta_standardized_rmse"] >= 0
        for key in ("selected_cap_fraction", "selected_dead_fraction",
                    "selected_relu_clip_fraction", "selected_zero_move_fraction"):
            assert 0 <= row[key] <= 1
    wrappers[1].reset_audit()
    assert wrappers[1].trace_records == []


def test_zero_dose_roundtrip_control_and_batch_accumulation():
    torch.manual_seed(17)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    classifier = TinyClassifier()
    with torch.no_grad():
        classifier.output.weight.zero_()
    wrapper = LatentClassifierGuidanceWrapper(
        sae, (5, 45), classifier, eta=1, diagnostic_trace=True)
    hidden = torch.randn(6, 4)
    targets = torch.tensor([[0, 1, 2], [1, 2, 3]])
    for _ in range(2):
        wrapper.set_targets(targets)
        with torch.no_grad():
            wrapper(hidden, torch.tensor([20, 0]))
    assert len(wrapper.trace_records) == 2
    with torch.no_grad():
        z = sae.encode(hidden[:3]).unsqueeze(0)
        expected = classifier(sae.encode(sae.decode(z))).softmax(-1).gather(
            -1, targets[:1].unsqueeze(-1)).squeeze(-1)[0]
    for row in wrapper.trace_records:
        assert row["batch_row"] == 0
        assert row["hidden_delta_rmse"] == 0
        assert row["latent_delta_rmse"] == 0
        assert row["roundtrip_latent_delta_rmse"] == 0
        assert row["probability_roundtrip_before"] == row["probability_roundtrip_after"]
        torch.testing.assert_close(torch.tensor(row["probability_roundtrip_before"]), expected)


def test_residual_preservation_is_identity_at_zero_gradient():
    torch.manual_seed(27)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    classifier = TinyClassifier()
    with torch.no_grad():
        classifier.output.weight.zero_()
    wrapper = LatentClassifierGuidanceWrapper(
        sae, (5, 45), classifier, eta=1, topk=2, topk_mode="dynamic",
        selection_score="applied", preserve_residual=True)
    hidden = torch.randn(6, 4)
    wrapper.set_targets(torch.tensor([[0, 1, 2], [1, 2, 3]]))
    with torch.no_grad():
        actual, _ = wrapper(hidden, torch.tensor([20, 0]))
    torch.testing.assert_close(actual, hidden)


def test_applied_selection_retains_largest_feasible_moves():
    torch.manual_seed(29)
    sae = TopKSparseAutoencoder(input_dim=4, latent_dim=8, topk_k=3)
    classifier = TinyClassifier()
    hidden = torch.randn(6, 4)
    wrappers = [LatentClassifierGuidanceWrapper(
        sae, (5, 45), classifier, eta=10, topk=k, topk_mode="dynamic",
        selection_score="applied") for k in (0, 2)]
    with torch.no_grad():
        base = sae.encode(hidden).reshape(2, 3, 8)
        outputs = []
        for wrapper in wrappers:
            wrapper.set_targets(torch.tensor([[0, 1, 2], [1, 2, 3]]))
            outputs.append(wrapper(hidden, torch.tensor([20, 20]))[1].reshape(2, 3, 8))
        dense_delta = outputs[0] - base
        idx = dense_delta.abs().topk(2, dim=-1).indices
        expected = torch.zeros_like(dense_delta).scatter(-1, idx, dense_delta.gather(-1, idx))
        torch.testing.assert_close(outputs[1] - base, expected)