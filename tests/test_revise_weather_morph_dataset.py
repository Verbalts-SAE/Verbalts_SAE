import numpy as np
import pytest

from scripts.revise_weather_morph_dataset import revise_arrays


def test_reduced_amplitude_and_reproducible_noise():
    backgrounds = np.zeros((2, 128, 1))
    residuals = np.ones((1000, 128, 1)) * 1.75
    records = [{"background_id": i % 2} for i in range(1000)]
    result = revise_arrays(backgrounds, residuals, records, 1.2 / 1.75, 0.06, 42)
    repeated = revise_arrays(backgrounds, residuals, records, 1.2 / 1.75, 0.06, 42)
    for a, b in zip(result, repeated):
        np.testing.assert_array_equal(a, b)
    samples, total, morphology, noise = result
    np.testing.assert_allclose(morphology, 1.2)
    np.testing.assert_allclose(samples, morphology + noise)
    np.testing.assert_array_equal(samples, total)
    assert abs(noise.mean()) < 0.001
    assert abs(noise.std() - 0.06) < 0.001
    clean = revise_arrays(backgrounds, residuals, records, 1, 0, 42)
    np.testing.assert_array_equal(clean[0], residuals)
    np.testing.assert_array_equal(residuals, 1.75)


@pytest.mark.parametrize("ratio,std", [(0, 0.06), (float("nan"), 0), (1, -1), (1, float("inf"))])
def test_invalid_parameters(ratio, std):
    with pytest.raises(ValueError):
        revise_arrays(np.zeros((1, 128, 1)), np.zeros((1, 128, 1)),
                      [{"background_id": 0}], ratio, std, 42)