import unittest

import numpy as np

from scripts.preview_cpu_adaptive_compression import compress_background


class AdaptiveCompressionTest(unittest.TestCase):
    def test_constant_and_small_smooth_unchanged(self):
        for x in (np.ones(128)*7, np.linspace(-.1, .1, 128)):
            y, m = compress_background(x)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(m['scale'], 1)

    def test_rough_background_compressed_more(self):
        smooth = np.linspace(-1, 1, 128)
        rough = np.tile([-1., 1.], 64)
        self.assertLess(compress_background(rough)[1]['scale'],
                        compress_background(smooth)[1]['scale'])

    def test_scalar_shape_and_translation(self):
        x = np.random.default_rng(42).normal(size=128)
        y, m = compress_background(x)
        np.testing.assert_allclose(np.diff(y), np.diff(x)*m['scale'], atol=1e-14)
        np.testing.assert_allclose(compress_background(x+10)[0], y+10)
        self.assertLessEqual(np.ptp(y), np.ptp(x))
        self.assertGreaterEqual(compress_background(x, budget_multiplier=1.25)[1]['scale'], m['scale'])

    def test_invalid(self):
        for x in ([1], [1, np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                compress_background(x)


if __name__ == '__main__':
    unittest.main()