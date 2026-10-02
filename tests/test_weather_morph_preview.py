import numpy as np
import pytest
from scipy.signal import find_peaks

from scripts.generate_weather_morph_preview import BOUNDS, NAMES, build, inject_weather


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("name,peaks,valleys", [("single_peak", 1, 0), ("double_peaks", 2, 1), ("sag", 0, 1)])
def test_weather_injection(stage, name, peaks, valleys):
    background = np.linspace(-2, 4, 128)
    labels = tuple(name if i == stage else "nothing" for i in range(4))
    curve, residual = inject_weather(background, labels)
    start, stop = BOUNDS[stage]
    np.testing.assert_array_equal(curve[:start], background[:start])
    np.testing.assert_array_equal(curve[stop:], background[stop:])
    np.testing.assert_allclose(curve - residual, background)
    assert residual[start] == residual[stop - 1] == 0
    assert len(find_peaks(residual[start:stop], prominence=0.12)[0]) == peaks
    assert len(find_peaks(-residual[start:stop], prominence=0.12)[0]) == valleys


def test_combinations_balanced_and_reproducible():
    from itertools import product
    labels = list(product(NAMES, repeat=4))
    backgrounds = np.zeros((6, 128))
    curves, residuals, records = build(backgrounds, labels, 1.75)
    assert curves.shape == (486, 128)
    np.testing.assert_array_equal(curves, build(backgrounds, labels, 1.75)[0])
    for stage in range(4):
        assert all(sum(r["morphologies"][stage] == n for r in records) == 162 for n in NAMES)
    np.testing.assert_array_equal(curves, residuals)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        inject_weather(np.zeros(128), NAMES)
    with pytest.raises(ValueError):
        inject_weather(np.zeros(128), ("sag",) * 4, amplitude=-1)