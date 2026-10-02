"""Create reproducible case-study plots for electricity_15min_morph v6.

The script reads only the finalized ``windows.npy`` and ``meta.csv``. Curves
are robust-normalized for shape comparison; their original scale remains
available in the metadata and is not modified.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/public/home/liym2024/Verbalts_SAE")
DEFAULT_DATASET = ROOT / "datasets/electricity_15min_morph"
TOPOLOGIES = (
    "single_peak",
    "single_trough",
    "peak_trough",
    "trough_peak",
    "oscillatory",
    "step_up",
    "step_down",
    "monotonic_up",
    "monotonic_down",
    "flat_complex",
)
PHASES = ("left", "center", "right")


def robust_normalize(windows: np.ndarray) -> np.ndarray:
    """Normalize each curve by its median and 5--95% range."""
    q05, q95 = np.quantile(windows, [0.05, 0.95], axis=1)
    scale = np.maximum(q95 - q05, 1e-8)
    return ((windows - np.median(windows, axis=1)[:, None]) / scale[:, None]).astype(
        np.float32
    )


def quantile_cases(indices: Iterable[int], distances: np.ndarray, count: int) -> np.ndarray:
    """Choose cases evenly across the empirical nearest-core distance ranking."""
    ordered = np.asarray(list(indices), dtype=np.int64)
    if not len(ordered):
        return ordered
    ordered = ordered[np.argsort(distances[ordered], kind="stable")]
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    return ordered[positions]


def annotate(ax, row: dict, distance: float, *, compact: bool = False) -> None:
    if compact:
        title = f"{row['channel_id']}  {row['selection_tier'][0].upper()}  d={distance:.3f}"
    else:
        title = (
            f"row {row['window_index']} | {row['channel_id']} | {row['selection_tier']}\n"
            f"p{row['prototype_id']} start={row['start_index']} d={distance:.3f}"
        )
    ax.set_title(title, fontsize=7)
    ax.set_xlim(0, 127)
    ax.set_xticks([0, 64, 127])
    ax.grid(alpha=0.18, linewidth=0.5)


def record_case(records: list[dict], figure: str, panel: str, row: dict) -> None:
    records.append(
        {
            "figure": figure,
            "panel": panel,
            "window_index": row["window_index"],
            "channel_id": row["channel_id"],
            "start_index": row["start_index"],
            "selection_tier": row["selection_tier"],
            "prototype_id": row["prototype_id"],
            "topology": row["topology"],
            "phase_bin": row["phase_bin"],
            "nearest_core_distance": row["nearest_core_distance"],
        }
    )


def plot_topology_cases(
    out: Path, curves: np.ndarray, rows: list[dict], distances: np.ndarray, records: list[dict]
) -> None:
    """Show each topology from near-duplicate-like to morphologically distant."""
    fig, axes = plt.subplots(len(TOPOLOGIES), 4, figsize=(15, 20), constrained_layout=True)
    labels = np.asarray([row["topology"] for row in rows])
    for ri, topology in enumerate(TOPOLOGIES):
        members = np.flatnonzero(labels == topology)
        chosen = quantile_cases(members, distances, 4)
        for ci, idx in enumerate(chosen):
            axes[ri, ci].plot(curves[idx], color="#2864a0", linewidth=1.0)
            annotate(axes[ri, ci], rows[idx], distances[idx], compact=True)
            record_case(records, "topology_distance_cases.png", f"{topology}:{ci}", rows[idx])
        for ci in range(len(chosen), 4):
            axes[ri, ci].axis("off")
        axes[ri, 0].set_ylabel(f"{topology}\n(n={len(members):,})", fontsize=9)
    fig.suptitle(
        "Topology cases across nearest-core distance (low → high)\n"
        "robust-normalized final noisy windows",
        fontsize=14,
    )
    fig.savefig(out / "topology_distance_cases.png", dpi=150)
    plt.close(fig)


def plot_phase_cases(
    out: Path, curves: np.ndarray, rows: list[dict], distances: np.ndarray, records: list[dict]
) -> None:
    """Compare left/center/right phases within topologies that support all phases."""
    labels = np.asarray([row["topology"] for row in rows])
    phases = np.asarray([row["phase_bin"] for row in rows])
    eligible = [
        topology
        for topology in TOPOLOGIES
        if all(np.any((labels == topology) & (phases == phase)) for phase in PHASES)
    ]
    fig, axes = plt.subplots(len(eligible), 3, figsize=(12, 2.7 * len(eligible)), constrained_layout=True)
    for ri, topology in enumerate(eligible):
        for ci, phase in enumerate(PHASES):
            members = np.flatnonzero((labels == topology) & (phases == phase))
            idx = int(quantile_cases(members, distances, 3)[1 if len(members) > 1 else 0])
            axes[ri, ci].plot(curves[idx], color=("#3b82b8", "#718096", "#d97732")[ci])
            annotate(axes[ri, ci], rows[idx], distances[idx], compact=True)
            if ri == 0:
                axes[ri, ci].set_title(
                    f"phase={phase}\n" + axes[ri, ci].get_title(), fontsize=8
                )
            record_case(records, "phase_cases.png", f"{topology}:{phase}", rows[idx])
        axes[ri, 0].set_ylabel(topology, fontsize=9)
    fig.suptitle("Matched-topology phase cases (median-distance examples)", fontsize=14)
    fig.savefig(out / "phase_cases.png", dpi=150)
    plt.close(fig)


def plot_prototype_cases(
    out: Path, curves: np.ndarray, rows: list[dict], distances: np.ndarray, records: list[dict]
) -> None:
    """Show a core anchor and expansion distance spectrum for six prototypes."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["prototype_id"]].append(idx)
    candidates = []
    for prototype, members in groups.items():
        core = [idx for idx in members if rows[idx]["selection_tier"] == "core"]
        expansion = [idx for idx in members if rows[idx]["selection_tier"] == "expansion"]
        if core and len(expansion) >= 3:
            candidates.append((len(members), int(prototype), core, expansion))
    candidates.sort(reverse=True)
    positions = np.linspace(0, len(candidates) - 1, 6).round().astype(int)
    selected = [candidates[pos] for pos in positions]

    fig, axes = plt.subplots(6, 4, figsize=(15, 15), constrained_layout=True)
    for ri, (_, prototype, core, expansion) in enumerate(selected):
        core_idx = int(quantile_cases(core, distances, 3)[len(quantile_cases(core, distances, 3)) // 2])
        chosen = [core_idx, *quantile_cases(expansion, distances, 3).tolist()]
        for ci, idx in enumerate(chosen):
            color = "#202020" if ci == 0 else ("#4c9a6a", "#d49a38", "#c44e52")[ci - 1]
            axes[ri, ci].plot(curves[idx], color=color)
            annotate(axes[ri, ci], rows[idx], distances[idx], compact=True)
            if ri == 0:
                heading = "core anchor" if ci == 0 else ("expansion low", "expansion mid", "expansion high")[ci - 1]
                axes[ri, ci].set_title(heading + "\n" + axes[ri, ci].get_title(), fontsize=8)
            record_case(records, "prototype_core_expansion_cases.png", f"p{prototype}:{ci}", rows[idx])
        axes[ri, 0].set_ylabel(f"prototype {prototype}\n{rows[core_idx]['topology']}", fontsize=9)
    fig.suptitle("Prototype cases: core anchor vs expansion distance spectrum", fontsize=14)
    fig.savefig(out / "prototype_core_expansion_cases.png", dpi=150)
    plt.close(fig)


def plot_channel_timeline_cases(
    out: Path, curves: np.ndarray, rows: list[dict], records: list[dict]
) -> None:
    """Show temporally ordered selected windows for topology-rich channels."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["channel_id"]].append(idx)
    ranked = sorted(
        groups.items(),
        key=lambda item: (len({rows[idx]["topology"] for idx in item[1]}), len(item[1]), item[0]),
        reverse=True,
    )[:6]
    fig, axes = plt.subplots(6, 8, figsize=(20, 14), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    topology_color = {name: palette(i) for i, name in enumerate(TOPOLOGIES)}
    for ri, (channel, members) in enumerate(ranked):
        ordered = sorted(members, key=lambda idx: int(rows[idx]["start_index"]))
        positions = np.linspace(0, len(ordered) - 1, 8).round().astype(int)
        chosen = [ordered[pos] for pos in positions]
        for ci, idx in enumerate(chosen):
            row = rows[idx]
            axes[ri, ci].plot(curves[idx], color=topology_color[row["topology"]], linewidth=1.0)
            axes[ri, ci].set_title(
                f"start={int(row['start_index']):,}\n{row['topology']} / {row['phase_bin']}", fontsize=7
            )
            axes[ri, ci].set_xlim(0, 127)
            axes[ri, ci].set_xticks([])
            axes[ri, ci].grid(alpha=0.15)
            record_case(records, "channel_timeline_cases.png", f"{channel}:{ci}", row)
        axes[ri, 0].set_ylabel(
            f"{channel}\n{len({rows[idx]['topology'] for idx in members})} topologies",
            fontsize=9,
        )
    fig.suptitle("Topology-rich channels: selected windows ordered by source start", fontsize=14)
    fig.savefig(out / "channel_timeline_cases.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve() if args.output else dataset / "visual_cases"
    windows = np.load(dataset / "windows.npy", mmap_mode="r")
    with (dataset / "meta.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if windows.ndim != 2 or windows.shape[1] != 128 or len(rows) != len(windows):
        raise ValueError(f"inconsistent dataset: windows={windows.shape}, metadata={len(rows)}")
    if any(int(row["window_index"]) != idx for idx, row in enumerate(rows)):
        raise ValueError("meta.csv window_index is not aligned with windows.npy")
    distances = np.asarray([float(row["nearest_core_distance"]) for row in rows])
    curves = robust_normalize(np.asarray(windows))

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    records: list[dict] = []
    plot_topology_cases(output, curves, rows, distances, records)
    plot_phase_cases(output, curves, rows, distances, records)
    plot_prototype_cases(output, curves, rows, distances, records)
    plot_channel_timeline_cases(output, curves, rows, records)
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(record["figure"] for record in records)
    print(f"saved {len(counts)} figures and {len(records)} indexed panels -> {output}")
    for name, count in counts.items():
        print(f"  {name}: {count} cases")


if __name__ == "__main__":
    main()