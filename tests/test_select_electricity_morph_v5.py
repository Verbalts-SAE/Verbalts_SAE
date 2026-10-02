import numpy as np

from scripts.select_electricity_morph_v5 import (
    align_profile,
    classify_profile,
    diversity_prune,
    shift_correlation,
)


def test_topology_peak_trough_and_monotonic():
    x = np.linspace(-1.0, 1.0, 32)
    peak = 1.5 * np.exp(-((x + 0.2) / 0.3) ** 2) - 0.3
    trough = -1.5 * np.exp(-((x - 0.2) / 0.3) ** 2) + 0.3
    assert classify_profile(peak)[0] == "single_peak"
    assert classify_profile(trough)[0] == "single_trough"
    assert classify_profile(x)[0] == "monotonic_up"
    assert classify_profile(-x)[0] == "monotonic_down"


def test_phase_bins_and_alignment_make_shifted_peaks_close():
    x = np.arange(32)
    left = np.exp(-((x - 8) / 3) ** 2)
    right = np.exp(-((x - 23) / 3) ** 2)
    left_label, left_phase, _ = classify_profile(left)
    right_label, right_phase, _ = classify_profile(right)
    assert left_label == right_label == "single_peak"
    assert left_phase == 0
    assert right_phase == 2
    a = align_profile(left, left_label, left_phase)
    b = align_profile(right, right_label, right_phase)
    assert np.sqrt(np.mean((a - b) ** 2)) < 0.05


def test_shift_correlation_recognizes_phase_duplicates():
    x = np.arange(128)
    a = np.sin(2 * np.pi * x / 64)
    b = np.sin(2 * np.pi * (x - 20) / 64)
    zero_lag = np.corrcoef(a, b)[0, 1]
    assert zero_lag < 0.0
    assert shift_correlation(a, b, max_shift=32) > 0.99


def test_diversity_prune_suppresses_duplicates_and_covers_channels():
    base = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    features = np.stack([base, base + 0.001, -base, np.sin(np.linspace(0, np.pi, 32))])
    candidates = {
        "canonical": features,
        "topology": np.zeros(4, dtype=np.int8),
        "channel": np.asarray([0, 0, 1, 2], dtype=np.int16),
        "relative_range": np.ones(4, dtype=np.float32),
        "repeated_jumps": np.zeros(4, dtype=np.int16),
    }
    selected = np.arange(4)
    kept, _, stats = diversity_prune(
        selected,
        np.asarray(["core"] * 4),
        candidates,
        np.arange(4),
        limit=4,
        radius=0.05,
    )
    assert len(set(kept) & {0, 1}) == 1
    assert set(candidates["channel"][kept]) == {0, 1, 2}
    assert stats["near_duplicate_or_limit_rejected"] == 1


def test_diversity_prune_rejects_isolated_profile_jump():
    smooth = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    jump = np.zeros(32, dtype=np.float32)
    jump[-1] = 3.0
    candidates = {
        "canonical": np.stack([smooth, jump]),
        "topology": np.zeros(2, dtype=np.int8),
        "channel": np.asarray([0, 1], dtype=np.int16),
        "relative_range": np.ones(2, dtype=np.float32),
        "repeated_jumps": np.zeros(2, dtype=np.int16),
    }
    kept, _, stats = diversity_prune(
        np.arange(2),
        np.asarray(["core", "core"]),
        candidates,
        np.arange(2),
        limit=2,
        radius=0.05,
    )
    assert kept.tolist() == [0]
    assert stats["quality_rejected"] == 1


def test_diversity_prune_rejects_only_low_amplitude_repeated_pulses():
    features = np.stack([
        np.sin(np.linspace(0, np.pi, 32)),
        np.sin(np.linspace(0, 2 * np.pi, 32)),
        np.cos(np.linspace(0, 2 * np.pi, 32)),
    ]).astype(np.float32)
    candidates = {
        "canonical": features,
        "topology": np.zeros(3, dtype=np.int8),
        "channel": np.arange(3, dtype=np.int16),
        "relative_range": np.asarray([0.004, 0.004, 0.20], dtype=np.float32),
        "repeated_jumps": np.asarray([20, 19, 40], dtype=np.int16),
    }
    kept, _, stats = diversity_prune(
        np.arange(3),
        np.asarray(["core"] * 3),
        candidates,
        np.arange(3),
        limit=3,
        radius=0.01,
    )
    assert set(kept) == {1, 2}
    assert stats["repeated_low_amplitude_pulse_rejected"] == 1