"""Tests for real-background SemiSynth-Morph generation primitives."""

import numpy as np
from scipy.signal import find_peaks

from contsg.data.datasets.semisynth_morph import (
    InjectionConfig,
    MORPHOLOGY_NAMES,
    STAGE_BOUNDS,
    balanced_combinations,
    inject,
    make_caption,
    morphology_residual,
    robust_normalize,
)
from sae.shapes import parse_segment_shapes


def test_balanced_combinations_are_deterministic_and_nearly_exact():
    first = balanced_combinations(2000, 42)
    second = balanced_combinations(2000, 42)
    np.testing.assert_array_equal(first, second)
    _, counts = np.unique(first, axis=0, return_counts=True)
    assert first.shape == (2000, 3)
    assert len(counts) == 64
    assert counts.max() - counts.min() <= 1


def test_residuals_are_endpoint_zero_and_have_expected_sign():
    assert InjectionConfig().amplitude == 1.75
    for name in MORPHOLOGY_NAMES:
        residual = morphology_residual(name, 43)
        assert residual[0] == residual[-1] == 0
        if name in {"single_peak", "double_peaks"}:
            assert residual.max() > 0
        elif name == "sag":
            assert residual.min() < 0
        else:
            assert not residual.any()


def test_inject_preserves_nothing_stages_and_boundaries():
    background = np.linspace(-2, 3, 128)
    curve, residual = inject(background, ("nothing", "single_peak", "nothing"))
    assert curve.shape == residual.shape == (128,)
    assert np.count_nonzero(residual[:43]) == 0
    assert np.count_nonzero(residual[85:]) == 0
    for boundary in (0, 42, 43, 85, 127):
        assert residual[boundary] == 0


def test_inject_adds_residual_directly_to_uncompressed_normalized_background():
    background = np.linspace(-2, 3, 128) ** 3
    normalized, _, _ = robust_normalize(background)
    curve, residual = inject(background, ("single_peak", "double_peaks", "sag"))
    np.testing.assert_allclose(curve, normalized + residual, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(normalized)) > 0.08


def test_inject_tanh_compresses_background_before_adding_residual():
    background = np.linspace(-2, 3, 128) ** 3
    normalized, _, _ = robust_normalize(background)
    scale = 2.0
    curve, residual = inject(
        background, ("single_peak", "double_peaks", "sag"), compression_scale=scale
    )
    np.testing.assert_allclose(curve, scale * np.tanh(normalized / scale) + residual,
                               rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(curve - residual)) <= scale


def test_caption_round_trips_through_project_parser():
    names = ("single_peak", "double_peaks", "sag")
    parsed = parse_segment_shapes(make_caption(names))
    assert parsed == ("single peak", "double peaks", "sag")


def test_injected_component_keeps_expected_shape_over_realistic_texture():
    background = 0.4 * np.sin(np.linspace(0, 5 * np.pi, 128)) + np.linspace(-1, 1, 128)
    normalized, _, _ = robust_normalize(background)
    for morphology, expected_peaks, expected_valleys in (
        ("single_peak", 1, 0),
        ("double_peaks", 2, 1),
        ("sag", 0, 1),
    ):
        curve, _ = inject(background, (morphology, "nothing", "nothing"))
        segment = curve[:43] - normalized[:43]
        peaks = find_peaks(segment, prominence=0.12, distance=4)[0]
        valleys = find_peaks(-segment, prominence=0.12, distance=4)[0]
        assert len(peaks) == expected_peaks
        assert len(valleys) == expected_valleys
