#!/usr/bin/env python3
"""Replot selected VerbalTS examples with complete three-stage conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sae.shapes import SHAPE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def add_boundaries(axis: plt.Axes) -> None:
    axis.axvline(42.5, color="0.45", linestyle=":", linewidth=0.9)
    axis.axvline(85.5, color="0.45", linestyle=":", linewidth=0.9)
    axis.grid(alpha=0.15, linewidth=0.5)


def condition_shapes(row: dict[str, object]) -> list[str]:
    return [SHAPE_NAMES[int(index)] for index in row["attrs_idx"]]


def annotation(row: dict[str, object], mse: float) -> tuple[str, list[bool]]:
    conditioned = condition_shapes(row)
    predicted = [str(shape) for shape in row["predicted_shapes"]]
    matches = [expected == actual for expected, actual in zip(conditioned, predicted)]
    title = (
        f"val idx {row['validation_index']} | MSE: {mse:.3f}\n"
        f"Condition (B/M/E): {' / '.join(conditioned)}\n"
        f"CNN pred. (B/M/E): {' / '.join(predicted)} | "
        f"full match: {'yes' if all(matches) else 'no'} | stage accuracy: {sum(matches)}/3"
    )
    return title, matches


def draw_example(
    axis: plt.Axes,
    reference: np.ndarray,
    generated: np.ndarray,
    row: dict[str, object],
    mse: float,
    *,
    title_size: float,
) -> list[bool]:
    x = np.arange(generated.shape[0])
    axis.plot(x, reference, color="0.55", linewidth=1.2, alpha=0.85, label="reference")
    axis.plot(x, generated, color="#1565c0", linewidth=1.4, label="VerbalTS")
    add_boundaries(axis)
    title, matches = annotation(row, mse)
    axis.set_title(title, fontsize=title_size)
    axis.set_xlabel("timestep")
    axis.set_ylabel("value")
    return matches


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = source_dir / "representative_examples"
    selection_path = output_dir / "examples.json"

    summary = json.loads((source_dir / "visualization_summary.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    generated = np.load(source_dir / "generated.npy")
    reference = np.load(source_dir / "reference.npy")

    enriched: list[dict[str, object]] = []
    selected_arrays: list[tuple[np.ndarray, np.ndarray, dict[str, object], float]] = []
    for old_example in selection["examples"]:
        source_row = int(old_example["source_row"])
        row = summary["rows"][source_row]
        mse = float(np.mean((generated[source_row] - reference[source_row]) ** 2))
        _, matches = annotation(row, mse)
        conditioned = condition_shapes(row)
        example = dict(old_example)
        example["conditioned_shapes"] = conditioned
        example["stage_matches"] = matches
        example["full_condition_match"] = all(matches)
        example["stage_accuracy"] = sum(matches) / len(matches)
        example["pointwise_mse"] = mse
        # Retain the legacy field for compatibility, but name its limited meaning clearly.
        target_index = ("beginning", "middle", "end").index(str(row["target_stage"]))
        example["target_stage_hit"] = matches[target_index]
        enriched.append(example)
        selected_arrays.append((reference[source_row], generated[source_row], row, mse))

    selection["selection_note"] = (
        "Representative hand-picked subset spanning low-to-high MSE and legacy "
        "single-target-stage hits/misses. Every condition is now reported for all three stages."
    )
    selection["condition_order"] = ["beginning", "middle", "end"]
    selection["match_note"] = (
        "full_condition_match requires all three CNN predictions to equal the three-stage condition; "
        "stage_accuracy is the matching-stage fraction. target_stage_hit is retained only for compatibility."
    )
    selection["examples"] = enriched
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(12.7, 10), constrained_layout=True)
    for axis, (truth, sample, row, mse) in zip(axes.flat, selected_arrays):
        draw_example(axis, truth, sample, row, mse, title_size=8.5)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Representative VerbalTS validation generations\n"
        "B/M/E = beginning/middle/end; conditions and CNN predictions cover every stage",
        fontsize=13,
    )
    fig.savefig(output_dir / "representative_examples.png", dpi=200)
    plt.close(fig)

    for truth, sample, row, mse in selected_arrays:
        fig, axis = plt.subplots(figsize=(8.2, 4.15), constrained_layout=True)
        draw_example(axis, truth, sample, row, mse, title_size=10)
        axis.legend(loc="upper left", fontsize=9)
        fig.savefig(output_dir / f"{row['concept']}.png", dpi=200)
        plt.close(fig)

    print(json.dumps({"output_dir": str(output_dir), "examples": len(enriched)}, indent=2))


if __name__ == "__main__":
    main()