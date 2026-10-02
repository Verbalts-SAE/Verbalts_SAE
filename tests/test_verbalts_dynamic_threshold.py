import pytest
import torch

from contsg.models.verbalts import VerbalTSModule


def test_dynamic_threshold_leaves_healthy_prediction_unchanged():
    prediction = torch.tensor([[[[-2.0, 0.0, 3.0]]]])
    actual = VerbalTSModule._apply_dynamic_threshold(
        prediction, data_scale=6.0, quantile=1.0
    )
    torch.testing.assert_close(actual, prediction)


def test_dynamic_threshold_rescales_unstable_prediction_per_sample():
    prediction = torch.tensor([
        [[[0.0, 4.0, 8.0]]],
        [[[0.0, 2.0, 3.0]]],
    ])
    actual = VerbalTSModule._apply_dynamic_threshold(
        prediction, data_scale=6.0, quantile=1.0
    )
    torch.testing.assert_close(actual[0], prediction[0] * 0.75)
    torch.testing.assert_close(actual[1], prediction[1])
    assert float(actual.abs().max()) == 6.0


@pytest.mark.parametrize(
    ("data_scale", "quantile"),
    [(0.0, 0.995), (-1.0, 0.995), (6.0, 0.0), (6.0, 1.01)],
)
def test_dynamic_threshold_rejects_invalid_parameters(data_scale, quantile):
    with pytest.raises(ValueError):
        VerbalTSModule._apply_dynamic_threshold(
            torch.ones(1, 1, 1, 4), data_scale=data_scale, quantile=quantile
        )