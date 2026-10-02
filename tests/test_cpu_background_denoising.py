import unittest

import numpy as np

from scripts.preview_cpu_background_denoising import denoise_background


class BackgroundDenoisingTest(unittest.TestCase):
    def test_constant_ramp_and_step_preserved(self):
        for x in (np.ones(128) * 7, np.linspace(-1, 1, 128), np.repeat([0., 1.], 64)):
            np.testing.assert_allclose(denoise_background(x), x, atol=1e-14)

    def test_spike_attenuation_and_mean(self):
        x = np.zeros(128)
        x[60] = 1
        light = denoise_background(x, .5)
        moderate = denoise_background(x, .25)
        self.assertAlmostEqual(np.ptp(light), .5)
        self.assertAlmostEqual(np.ptp(moderate), .25)
        self.assertAlmostEqual(moderate.mean(), x.mean())

    def test_identity_translation_and_residual(self):
        x = np.random.default_rng(42).normal(size=128)
        np.testing.assert_array_equal(denoise_background(x, 1), x)
        y = denoise_background(x)
        np.testing.assert_allclose(denoise_background(x + 10), y + 10)
        r = np.zeros(128)
        r[50:55] = [0, .6, 1.2, .6, 0]
        np.testing.assert_allclose((y + r) - y, r, atol=1e-14)

    def test_invalid_inputs(self):
        for x in ([1], [1, np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                denoise_background(x)
        for retention in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                denoise_background(np.ones(10), retention)
        for window in (0, 2, 11):
            with self.assertRaises(ValueError):
                denoise_background(np.ones(10), window=window)


if __name__ == "__main__":
    unittest.main()