from copy import deepcopy

import numpy as np

from scripts.preview_weather_relaxed_scale import rescale_background


def test_rescale_preserves_steps_joins_and_source():
    background = np.array([0., 0.3, 0.3, 0.4, 0.4, 1.4])
    record = {"segments": [
        dict(output_start=0, output_stop=2, scale=0.3, channel="temperature",
             max_step_ratio=0.1, total_variation_ratio=2),
        dict(output_start=2, output_stop=4, scale=0.1, channel="wind_speed",
             max_step_ratio=0.5, total_variation_ratio=3),
        dict(output_start=4, output_stop=6, scale=1., channel="air_pressure",
             max_step_ratio=0.1, total_variation_ratio=2),
    ]}
    original = deepcopy(record)
    result, updated = rescale_background(background, record)
    np.testing.assert_allclose(result, [0, 0.4, 0.4, 0.6, 0.6, 1.6])
    assert record == original
    assert [s["scale"] for s in updated["segments"]] == [0.4, 0.2, 1.]
    np.testing.assert_array_equal(background, [0, 0.3, 0.3, 0.4, 0.4, 1.4])