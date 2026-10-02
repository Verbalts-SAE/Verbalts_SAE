"""Interpretable tsfresh global features and dataset-relative captions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features


FEATURE_NAMES = (
    "linear_trend_slope",
    "standard_deviation",
    "mean_abs_change",
    "cid_ce_normalized",
    "autocorrelation_lag_1",
)

# Every numeric feature is converted to a dataset-relative categorical phrase;
# raw values and thresholds are never included in captions.
CAPTION_FEATURE_NAMES = FEATURE_NAMES

TSFRESH_SETTINGS = {
    "linear_trend": [{"attr": "slope"}],
    "standard_deviation": None,
    "mean_abs_change": None,
    "cid_ce": [{"normalize": True}],
    "autocorrelation": [{"lag": 1}],
}

_TSFRESH_COLUMNS = (
    'value__linear_trend__attr_"slope"',
    "value__standard_deviation",
    "value__mean_abs_change",
    "value__cid_ce__normalize_True",
    "value__autocorrelation__lag_1",
)


def extract_global_features(curves: np.ndarray, n_jobs: int = 0) -> np.ndarray:
    """Extract five interpretable whole-curve tsfresh features.

    Args:
        curves: Array with shape ``(samples, length)`` or ``(samples, length, 1)``.
        n_jobs: Number of tsfresh worker processes; zero uses all available CPUs.
    """

    values = np.asarray(curves)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"expected curves with shape (N, L[, 1]), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("curves contain non-finite values")
    if len(values) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    sample_count, length = values.shape
    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(sample_count), length),
            "time": np.tile(np.arange(length), sample_count),
            "value": values.reshape(-1),
        }
    )
    extracted = extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=TSFRESH_SETTINGS,
        n_jobs=n_jobs,
        disable_progressbar=True,
    ).sort_index()
    features = extracted.loc[:, _TSFRESH_COLUMNS].to_numpy(dtype=np.float32)
    if features.shape != (sample_count, len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("tsfresh returned incomplete or non-finite global features")
    return features


def fit_caption_thresholds(features: np.ndarray) -> dict[str, tuple[float, float]]:
    """Fit train-only tertile thresholds used to verbalize each feature."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) == 0:
        raise ValueError(f"expected a non-empty (N, {len(FEATURE_NAMES)}) feature matrix")
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    quantiles = np.quantile(values, (1 / 3, 2 / 3), axis=0)
    return {
        name: (float(quantiles[0, index]), float(quantiles[1, index]))
        for index, name in enumerate(FEATURE_NAMES)
    }


def _level(value: float, bounds: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if value < bounds[0]:
        return labels[0]
    if value > bounds[1]:
        return labels[2]
    return labels[1]


def make_global_description(
    features: np.ndarray,
    thresholds: Mapping[str, tuple[float, float]],
) -> str:
    """Turn one tsfresh row into a compact global English description."""

    row = np.asarray(features, dtype=np.float64)
    if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
        raise ValueError(f"expected one finite feature row with shape ({len(FEATURE_NAMES)},)")
    missing = set(CAPTION_FEATURE_NAMES) - set(thresholds)
    if missing:
        raise ValueError(f"missing caption thresholds for: {sorted(missing)}")

    trend = _level(
        row[0], thresholds["linear_trend_slope"], ("a downward", "a mostly level", "an upward")
    )
    variability = _level(
        row[1], thresholds["standard_deviation"], ("low", "moderate", "high")
    )
    changes = _level(
        row[2], thresholds["mean_abs_change"], ("smooth", "moderate", "rapid")
    )
    complexity = _level(
        row[3], thresholds["cid_ce_normalized"], ("low", "moderate", "high")
    )
    persistence = _level(
        row[4], thresholds["autocorrelation_lag_1"], ("weak", "moderate", "strong")
    )
    return (
        f"The time series has {trend} trend overall. "
        f"Its global pattern has {variability} variability, {changes} point-to-point changes, "
        f"{complexity} structural complexity, and {persistence} short-term persistence."
    )


def combine_global_and_local_caption(global_description: str, local_caption: str) -> str:
    """Place the global summary before the existing fine-grained description."""

    readable_local_caption = local_caption.strip().replace(".The ", ". The ")
    return f"{global_description.strip()} {readable_local_caption}"