"""Render representative SemiSynth-Morph cases and injection details."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.data.datasets.semisynth_morph import (  # noqa: E402
    MORPHOLOGY_NAMES,
    STAGE_BOUNDS,
    robust_normalize,
)

DEFAULT_SOURCE = ROOT / "datasets" / "electricity_15min_morph"
DEFAULT_DATASET = ROOT / "datasets" / "electricity_15min_semisynth_morph"

# Includes an empty control, isolated events, repeated events, and mixed orderings.
SELECTED_COMBINATIONS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 2, 0),
    (0, 0, 3),
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3),
    (1, 2, 3),
    (2, 3, 1),
    (3, 1, 2),
    (1, 3, 2),
    (3, 2, 1),
)

COLORS = {
    "nothing": "#7f8c8d",
    "single_peak": "#d95f02",
    "double_peaks": "#1b9e77",
    "sag": "#7570b3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_cases(source: Path, dataset: Path, split: str) -> list[dict]:
    windows = np.load(source / "windows.npy", mmap_mode="r")
    curves = np.load(dataset / f"{split}_ts.npy", mmap_mode="r")
    attrs = np.load(dataset / f"{split}_attrs_idx.npy", mmap_mode="r")
    source_indices = np.load(dataset / f"{split}_source_indices.npy")
    captions = np.load(dataset / f"{split}_text_caps.npy")
    with (source / "meta.csv").open(newline="", encoding="utf-8") as handle:
        source_meta = list(csv.DictReader(handle))

    cases = []
    for combination in SELECTED_COMBINATIONS:
        matches = np.flatnonzero(np.all(attrs == np.asarray(combination), axis=1))
        if not len(matches):
            raise ValueError(f"combination {combination} is absent from split {split}")
        index = int(matches[0])
        source_index = int(source_indices[index])
        raw = np.asarray(windows[source_index], dtype=np.float64)
        normalized, median, iqr = robust_normalize(raw)
        base = normalized
        final = np.asarray(curves[index, :, 0], dtype=np.float64)
        names = tuple(MORPHOLOGY_NAMES[value] for value in combination)
        cases.append(
            {
                "split": split,
                "split_index": index,
                "source_index": source_index,
                "source_meta": source_meta[source_index],
                "raw": raw,
                "base": base,
                "residual": final - base,
                "final": final,
                "attrs": combination,
                "names": names,
                "caption": str(np.asarray(captions[index]).reshape(-1)[0]),
                "median": median,
                "iqr": iqr,
            }
        )
    return cases


def add_stage_context(axis: plt.Axes, names: tuple[str, str, str]) -> None:
    axis.axvline(42.5, color="0.45", lw=0.7, ls=":")
    axis.axvline(85.5, color="0.45", lw=0.7, ls=":")
    for name, (start, stop) in zip(names, STAGE_BOUNDS):
        axis.axvspan(start, stop - 1, color=COLORS[name], alpha=0.045, lw=0)
    axis.set_xlim(0, 127)
    axis.grid(axis="y", color="0.9", lw=0.5)


def short_names(names: tuple[str, str, str]) -> str:
    aliases = {"nothing": "none", "single_peak": "single", "double_peaks": "double", "sag": "sag"}
    return " / ".join(aliases[name] for name in names)


def plot_overview(cases: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True, sharey=True)
    x = np.arange(128)
    for case, axis in zip(cases, axes.flat):
        add_stage_context(axis, case["names"])
        axis.plot(x, case["base"], color="0.55", lw=0.9, ls="--", label="normalized real background")
        axis.plot(x, case["final"], color="#2166ac", lw=1.7, label="semi-synthetic curve")
        axis.fill_between(x, case["base"], case["final"], color="#67a9cf", alpha=0.18)
        axis.set_title(
            f'{case["split"]}[{case["split_index"]}]  {short_names(case["names"])}',
            fontsize=9,
        )
    for axis in axes[-1]:
        axis.set_xlabel("time step")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized value")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.suptitle("Representative SemiSynth-Morph cases (dashed = normalized real background)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "representative_cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_details(cases: list[dict], output: Path) -> None:
    # Six examples are enough for readable component-level comparison.
    detail_cases = [cases[i] for i in (0, 1, 2, 3, 7, 8)]
    fig, axes = plt.subplots(len(detail_cases), 3, figsize=(16, 15))
    x = np.arange(128)
    for row, case in enumerate(detail_cases):
        raw_axis, component_axis, final_axis = axes[row]
        raw_axis.plot(x, case["raw"], color="#4d4d4d", lw=1.1)
        raw_axis.set_title(
            f'Raw source: {case["source_meta"]["channel_id"]}, window {case["source_index"]}',
            fontsize=9,
        )
        raw_axis.set_ylabel("original units")
        raw_axis.grid(axis="y", color="0.9", lw=0.5)

        add_stage_context(component_axis, case["names"])
        component_axis.plot(x, case["base"], color="0.45", lw=1.0, label="normalized real background")
        component_axis.plot(x, case["residual"], color="#d95f02", lw=1.25, label="injected residual")
        component_axis.axhline(0, color="0.75", lw=0.6)
        component_axis.set_title("Components", fontsize=9)

        add_stage_context(final_axis, case["names"])
        final_axis.plot(x, case["final"], color="#2166ac", lw=1.6)
        final_axis.set_title(f'Final: {short_names(case["names"])}', fontsize=9)
        final_axis.set_ylabel("normalized value")
        if row == 0:
            component_axis.legend(loc="upper right", fontsize=7, frameon=False)
        if row == len(detail_cases) - 1:
            for axis in (raw_axis, component_axis, final_axis):
                axis.set_xlabel("time step")
    fig.suptitle("Injection details: real electricity background, components, and final curve", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output / "injection_details.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_case_index(cases: list[dict], output: Path) -> None:
    fields = (
        "split",
        "split_index",
        "source_window_index",
        "channel_id",
        "source_start_index",
        "beginning",
        "middle",
        "end",
        "source_median",
        "source_iqr",
        "caption",
    )
    with (output / "selected_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "split": case["split"],
                    "split_index": case["split_index"],
                    "source_window_index": case["source_index"],
                    "channel_id": case["source_meta"]["channel_id"],
                    "source_start_index": case["source_meta"]["start_index"],
                    "beginning": case["names"][0],
                    "middle": case["names"][1],
                    "end": case["names"][2],
                    "source_median": case["median"],
                    "source_iqr": case["iqr"],
                    "caption": case["caption"],
                }
            )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve() if args.output else dataset / "examples"
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(source, dataset, args.split)
    plot_overview(cases, output)
    plot_details(cases, output)
    write_case_index(cases, output)
    print(f"Rendered {len(cases)} representative cases to {output}")


if __name__ == "__main__":
    main()