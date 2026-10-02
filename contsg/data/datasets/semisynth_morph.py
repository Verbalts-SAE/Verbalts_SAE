"""Deterministic morphology injection on normalized real-world backgrounds.

The resulting curves retain the ordering and local texture of each real
background while making the supervised local morphology unambiguous.  Every
non-empty residual is smooth and exactly zero at its segment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


MORPHOLOGY_NAMES = ("nothing", "single_peak", "double_peaks", "sag")
STAGE_NAMES = ("beginning", "middle", "end")
STAGE_BOUNDS = ((0, 43), (43, 86), (85, 128))


@Registry.register_dataset(
    "electricity-semisynth-morph",
    aliases=["electricity_15min_semisynth_morph"],
)
class ElectricitySemiSynthMorphDataModule(BaseDataModule):
    """Three-stage morphology injection on held-out electricity backgrounds."""


@dataclass(frozen=True)
class InjectionConfig:
    """Controls local morphology shape and strength."""

    amplitude: float = 1.75
    width: float = 0.08
    double_separation: float = 0.34


def robust_normalize(curve: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Median/IQR normalize one finite length-128 curve."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (128,) or not np.isfinite(curve).all():
        raise ValueError("curve must be a finite array with shape (128,)")
    median = float(np.median(curve))
    q25, q75 = np.quantile(curve, (0.25, 0.75))
    scale = float(q75 - q25)
    if scale < 1e-6:
        scale = float(np.std(curve))
    if scale < 1e-6:
        scale = 1.0
    return ((curve - median) / scale).astype(np.float64), median, scale


def morphology_residual(
    morphology: str,
    length: int,
    config: InjectionConfig = InjectionConfig(),
) -> np.ndarray:
    """Build an endpoint-zero shapelet for one segment."""
    if morphology not in MORPHOLOGY_NAMES:
        raise ValueError(f"unknown morphology: {morphology}")
    if length < 5:
        raise ValueError("segment length must be at least 5")
    if morphology == "nothing":
        return np.zeros(length, dtype=np.float64)

    x = np.linspace(0.0, 1.0, length)
    envelope = np.sin(np.pi * x) ** 2

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - center) / config.width) ** 2)

    if morphology == "single_peak":
        shape = gaussian(0.5)
    elif morphology == "double_peaks":
        offset = config.double_separation / 2.0
        shape = gaussian(0.5 - offset) + gaussian(0.5 + offset)
        # A central notch keeps both maxima identifiable over real texture.
        shape -= 0.60 * gaussian(0.5)
        shape /= shape.max()
    else:  # sag
        shape = -gaussian(0.5)

    residual = config.amplitude * envelope * shape
    residual[[0, -1]] = 0.0
    return residual


def inject(
    background: np.ndarray,
    morphologies: tuple[str, str, str],
    config: InjectionConfig = InjectionConfig(),
    compression_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize, optionally tanh-compress, then inject three stage labels."""
    if len(morphologies) != 3:
        raise ValueError("exactly three stage morphologies are required")
    base, _, _ = robust_normalize(background)
    if compression_scale is not None:
        if not np.isfinite(compression_scale) or compression_scale <= 0:
            raise ValueError("compression_scale must be finite and positive")
        base = compression_scale * np.tanh(base / compression_scale)
    residual = np.zeros(128, dtype=np.float64)
    for morphology, (start, stop) in zip(morphologies, STAGE_BOUNDS):
        # Stages 2 and 3 overlap at t=85. Both residuals are zero there.
        residual[start:stop] += morphology_residual(morphology, stop - start, config)
    return (base + residual).astype(np.float32), residual.astype(np.float32)


def balanced_combinations(size: int, seed: int) -> np.ndarray:
    """Return a deterministic near-exact balance over all 4^3 label tuples."""
    if size < 0:
        raise ValueError("size must be non-negative")
    combinations = np.asarray(list(product(range(4), repeat=3)), dtype=np.int64)
    labels = np.tile(combinations, (size // len(combinations), 1))
    remainder = size % len(combinations)
    rng = np.random.default_rng(seed)
    if remainder:
        labels = np.concatenate((labels, combinations[rng.permutation(64)[:remainder]]))
    return labels[rng.permutation(size)] if size else labels


def make_caption(morphologies: tuple[str, str, str]) -> str:
    """Create a caption accepted by :func:`sae.shapes.parse_segment_shapes`."""
    display = {
        "nothing": "nothing",
        "single_peak": "a single peak",
        "double_peaks": "double peaks",
        "sag": "a sag",
    }
    return "".join(
        f"The {stage} part has {display[morphology]}."
        for stage, morphology in zip(STAGE_NAMES, morphologies)
    )
