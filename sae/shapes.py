"""Shared shape vocabulary and parsing utilities for synth-u.

Defines the 3 stage names and 4 shape names used by every downstream
component (classifier training, guidance evaluation, visualization),
plus caption parsing, segment splitting, and CNN curve annotation helpers.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D


STAGE_NAMES = ("Beginning", "Middle", "End")
SHAPE_NAMES = ("nothing", "single peak", "double peaks", "sag")
SHAPE_TO_TARGET = {
    "nothing": (0, 0),
    "single peak": (1, 0),
    "double peaks": (2, 0),
    "sag": (0, 1),
}


def parse_segment_shapes(caption: str) -> tuple[str, str, str]:
    """Extract beginning/middle/end local shapes from a synth-u caption.

    The retained dataset uses exactly three equivalent phrasings for each
    stage/shape combination (for example, ``A sag at beginning area`` and
    ``The beginning part has a sag``).  Global trend/season statements are
    ignored.  If a stage has no local statement, its target is ``nothing``.
    """

    shapes = ["nothing", "nothing", "nothing"]
    found: dict[int, str] = {}

    for raw_sentence in re.split(r"[.\n]", str(caption)):
        sentence = raw_sentence.strip().lower()
        if not sentence:
            continue

        stage: int | None = None
        if "beginning" in sentence or "at beginning" in sentence:
            stage = 0
        elif "middle" in sentence:
            stage = 1
        elif "end part" in sentence or "end area" in sentence or "at end" in sentence:
            stage = 2
        if stage is None:
            continue

        shape: str | None = None
        if "double peak" in sentence:
            shape = "double peaks"
        elif "single peak" in sentence:
            shape = "single peak"
        elif "sag" in sentence:
            shape = "sag"
        if shape is None:
            raise ValueError(f"unrecognized local-shape statement: {raw_sentence!r}")
        if stage in found and found[stage] != shape:
            raise ValueError(
                f"conflicting labels for {STAGE_NAMES[stage]}: "
                f"{found[stage]!r} and {shape!r} in {caption!r}"
            )
        found[stage] = shape
        shapes[stage] = shape

    return tuple(shapes)  # type: ignore[return-value]


def split_segments(curves: np.ndarray) -> np.ndarray:
    """Split (N, 128) curves into the three 43-step CNN inputs.

    This preserves the boundary convention of the original classifier data:
    ``[0:43]``, ``[43:86]``, and ``[85:128]``.  The final two segments overlap
    at timestep 85 by one point.
    """

    curves = np.asarray(curves)
    if curves.ndim == 3 and curves.shape[-1] == 1:
        curves = curves[..., 0]
    if curves.ndim != 2 or curves.shape[1] != 128:
        raise ValueError(f"expected curves with shape (N, 128[, 1]), got {curves.shape}")
    return np.stack((curves[:, :43], curves[:, 43:86], curves[:, 85:128]), axis=1)


def captions_to_targets(captions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert captions to shape names and the CNN's two target heads."""

    names = np.asarray(
        [parse_segment_shapes(str(caption)) for caption in captions.reshape(-1)],
        dtype="U16",
    )
    peak = np.empty(names.shape, dtype=np.int64)
    valley = np.empty(names.shape, dtype=np.int64)
    for shape, target in SHAPE_TO_TARGET.items():
        selected = names == shape
        peak[selected] = target[0]
        valley[selected] = target[1]
    return names, peak, valley


def _shape_from_heads(peak: int, valley: int) -> str:
    if valley == 1 and peak == 0:
        return "sag"
    if valley == 0 and peak == 0:
        return "nothing"
    if valley == 0 and peak == 1:
        return "single peak"
    if valley == 0 and peak == 2:
        return "double peaks"
    peak_name = "single peak" if peak == 1 else "double peaks"
    return f"{peak_name} + sag"


@torch.no_grad()


def classify_curves(
    model: PeakValleyClassifier1D,
    curves: np.ndarray,
    device: torch.device,
) -> list[list[dict[str, Any]]]:
    """Classify each segment and retain both head probabilities."""

    model.eval()
    segments = split_segments(curves)
    n_curves = segments.shape[0]
    inputs = torch.from_numpy(segments.reshape(-1, 1, 43).astype(np.float32)).to(device)
    peak_logits, valley_logits = model(inputs)
    peak_prob = peak_logits.softmax(dim=1).cpu().numpy()
    valley_prob = valley_logits.softmax(dim=1).cpu().numpy()
    peak_pred = peak_prob.argmax(axis=1)
    valley_pred = valley_prob.argmax(axis=1)

    predictions: list[list[dict[str, Any]]] = []
    for curve_index in range(n_curves):
        curve_predictions = []
        for stage in range(3):
            flat_index = curve_index * 3 + stage
            peak = int(peak_pred[flat_index])
            valley = int(valley_pred[flat_index])
            curve_predictions.append(
                {
                    "shape": _shape_from_heads(peak, valley),
                    "peak_count": peak,
                    "valley_count": valley,
                    "peak_probabilities": peak_prob[flat_index].tolist(),
                    "valley_probabilities": valley_prob[flat_index].tolist(),
                    "joint_confidence": float(
                        peak_prob[flat_index, peak] * valley_prob[flat_index, valley]
                    ),
                }
            )
        predictions.append(curve_predictions)
    return predictions
