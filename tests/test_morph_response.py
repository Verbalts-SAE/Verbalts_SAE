import torch
import numpy as np
import pytest
from torch import nn

from sae.diagnose_morph_response import CaptureTransform, rmse_rows, guidance_steps, probability_metrics


def test_capture_is_observational():
    class Shift(nn.Module):
        def forward(self, hidden, step):
            return hidden + 2, hidden * 3

    hidden = torch.arange(12.).reshape(4, 3)
    observer = CaptureTransform(Shift())
    output, latent = observer(hidden, torch.tensor([25]))
    assert torch.equal(output, hidden + 2)
    assert torch.equal(latent, hidden * 3)
    assert torch.equal(observer.before, hidden)
    assert torch.equal(observer.after, output)
    output.zero_()
    assert torch.equal(observer.after, hidden + 2)


def test_rmse_preserves_sample_axis():
    a = torch.tensor([[[3., 4.]], [[0., 0.]]])
    torch.testing.assert_close(rmse_rows(a, torch.zeros_like(a)), torch.tensor([12.5 ** .5, 0.]))


def test_window_schedule():
    assert guidance_steps(40, 1, (5, 45)) == [40]
    assert guidance_steps(10, 5, (5, 45)) == [10, 9, 8, 7, 6]
    for start, length in [(5, 2), (46, 1), (25, 0)]:
        with pytest.raises(ValueError):
            guidance_steps(start, length, (5, 45))


def test_probability_metrics_direction_and_empty_group():
    baseline = [[dict(shape='nothing', peak_probabilities=[.6, .3, .1],
                      valley_probabilities=[.8, .2]) for _ in range(3)]]
    changed = [[dict(shape='nothing', peak_probabilities=[.5, .4, .1],
                     valley_probabilities=[.7, .3]) for _ in range(3)]]
    names = np.array([['single peak', 'sag', 'nothing']])
    result = probability_metrics(changed, baseline, names)
    np.testing.assert_allclose(result['mean_delta_per_stage'], [.04, .03, -.13])
    assert result['improved_per_stage'] == [1, 1, 0]
    assert result['worsened_per_stage'] == [0, 0, 1]
    assert result['baseline_incorrect_count_per_stage'] == [1, 1, 0]
    assert result['mean_delta_baseline_incorrect_per_stage'][2] is None
    identity = probability_metrics(baseline, baseline, names)
    assert identity['mean_delta_per_stage'] == [0., 0., 0.]