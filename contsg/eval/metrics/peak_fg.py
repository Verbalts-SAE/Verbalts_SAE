"""Analytic three-stage adherence metric for Synth-Peak-FG."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from contsg.eval.metrics.base import CollectiveMetric
from contsg.registry import Registry

PEAK_SPANS = (5, 7, 9, 11, 15, 17)
STAGE_NAMES = ("early", "middle", "late")
STAGE_BOUNDS = ((0, 42), (42, 85), (85, 128))


def _width_class(span: int) -> int:
    if span in (5, 7):
        return 0
    if span in (9, 11):
        return 1
    if span in (15, 17):
        return 2
    raise ValueError(f"unsupported peak span: {span}")


def _height_class(prominence: float) -> int:
    return 0 if prominence < 0.90 else 1 if prominence < 1.30 else 2


def detect_peak_parameters(
    signal: np.ndarray,
    center_bounds: tuple[int, int] | None = None,
) -> tuple[int, int, float, float]:
    """Return the strongest triangle's center, span, prominence, and score."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    lower, upper = center_bounds or (0, values.size)
    best = (-1, -1, 0.0, -np.inf)
    for span in PEAK_SPANS:
        template = np.bartlett(span + 2).astype(np.float64)
        norm_sq = float(template @ template)
        half = (span + 1) // 2
        windows = np.lib.stride_tricks.sliding_window_view(values, span + 2)
        fractions = np.linspace(0.0, 1.0, span + 2)
        residuals = windows - (
            windows[:, :1] + (windows[:, -1:] - windows[:, :1]) * fractions
        )
        projections = residuals @ template
        amplitudes = projections / norm_sq
        denominators = np.linalg.norm(residuals, axis=1) * np.sqrt(norm_sq)
        correlations = np.divide(
            projections,
            denominators,
            out=np.zeros_like(projections),
            where=denominators > 0,
        )
        scores = amplitudes * np.clip(correlations, 0, 1)
        scores[amplitudes <= 0] = 0
        centers = np.arange(len(scores)) + half
        allowed = (centers >= lower) & (centers < upper)
        if not np.any(allowed):
            continue
        allowed_indices = np.flatnonzero(allowed)
        index = int(allowed_indices[np.argmax(scores[allowed])])
        score = float(scores[index])
        if score > best[3]:
            best = (int(centers[index]), span, float(np.max(residuals[index])), score)
    return best[0], best[1], best[2], max(0.0, best[3])


@Registry.register_metric("peak_fg_acc")
class PeakFGAccuracyMetric(CollectiveMetric):
    """Measure each 9-way stage type and all-three joint adherence."""

    def __init__(self, name: str = "peak_fg_acc", score_min: float = 0.40) -> None:
        super().__init__(name)
        self.score_min = float(score_min)
        self._pred_list: List[np.ndarray] = []
        self._attrs_list: List[np.ndarray] = []

    def reset(self) -> None:
        self._pred_list = []
        self._attrs_list = []

    def update(self, batch_data: Dict[str, Any]) -> None:
        if "pred" not in batch_data:
            raise KeyError("peak_fg_acc requires 'pred' in batch_data")
        attrs = batch_data.get("attrs", batch_data.get("attrs_idx"))
        if attrs is None:
            raise KeyError("peak_fg_acc requires 'attrs' or 'attrs_idx'")
        pred = batch_data["pred"]
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(attrs, torch.Tensor):
            attrs = attrs.detach().cpu().numpy()
        pred_array, attrs_array = np.asarray(pred), np.asarray(attrs)
        if pred_array.ndim != 3 or pred_array.shape[-1] != 1:
            raise ValueError(f"pred must have shape (B, L, 1), got {pred_array.shape}")
        if attrs_array.ndim != 2 or attrs_array.shape[1] < 6:
            raise ValueError(f"attrs must have at least six columns, got {attrs_array.shape}")
        self._pred_list.append(pred_array[..., 0])
        self._attrs_list.append(attrs_array)

    def compute(self) -> Dict[str, float]:
        if not self._pred_list:
            return {
                **{f"{stage}_accuracy": 0.0 for stage in STAGE_NAMES},
                "mean_stage_accuracy": 0.0,
                "joint_accuracy": 0.0,
                "all_stage_event_rate": 0.0,
            }
        predictions = np.concatenate(self._pred_list)
        attrs = np.concatenate(self._attrs_list)
        hits = np.zeros((len(predictions), 3), dtype=bool)
        events = np.zeros_like(hits)
        for sample, (signal, attr) in enumerate(zip(predictions, attrs)):
            for stage, bounds in enumerate(STAGE_BOUNDS):
                _, span, prominence, score = detect_peak_parameters(signal, bounds)
                events[sample, stage] = score >= self.score_min
                predicted_type = 3 * _width_class(span) + _height_class(prominence)
                hits[sample, stage] = events[sample, stage] and predicted_type == int(attr[3 + stage])
        return {
            **{
                f"{name}_accuracy": float(hits[:, stage].mean())
                for stage, name in enumerate(STAGE_NAMES)
            },
            "mean_stage_accuracy": float(hits.mean()),
            "joint_accuracy": float(hits.all(axis=1).mean()),
            "all_stage_event_rate": float(events.all(axis=1).mean()),
        }


__all__ = ["PeakFGAccuracyMetric", "detect_peak_parameters"]