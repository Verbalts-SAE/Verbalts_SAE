"""Sample diverse 128-point windows across channels in real Arrow datasets.

Each Arrow row is treated as one univariate channel.  A returned sample is one
contiguous window from one channel (shape [128]); diversity is obtained by
sampling across channel IDs and time positions, not by concatenating channels.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa


ROOT = Path("/public/home/liym2024/Verbalts_SAE")
ARROW_ROOT = Path("/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow")
OUT = ROOT / "results/real_datasets_128_channel_diverse"
DATASETS = [
    "monash_electricity_hourly",
    "monash_fred_md",
    "exchange_rate",
    "ercot",
    "electricity_15min",
]
WINDOW = 128
N_SAMPLES = 100
SEED = 42

DOMAIN = {
    "monash_electricity_hourly": ("hourly electricity-consumption", "hours"),
    "monash_fred_md": ("monthly macroeconomic-indicator", "months"),
    "exchange_rate": ("daily exchange-rate", "days"),
    "ercot": ("hourly ERCOT electricity-load", "hours"),
    "electricity_15min": ("15-minute electricity-consumption", "hours"),
}


def load_dataset(name: str):
    path = ARROW_ROOT / name / "train-00000-of-00001.arrow"
    with pa.ipc.open_stream(str(path)) as reader:
        value_col = next(c for c in reader.schema.names if c not in {"id", "timestamp"})
        rows = []
        for batch in reader:
            ids = batch["id"].to_pylist()
            timestamps = batch["timestamp"].to_pylist()
            values = batch[value_col].to_pylist()
            rows.extend(
                {"id": i, "timestamp": t, "values": np.asarray(v, dtype=np.float64)}
                for i, t, v in zip(ids, timestamps, values)
            )
    return rows, value_col


def adaptive_stride(length: int) -> int:
    """Use small stride when only a few windows fit, larger stride for long data."""
    available = max(0, length - WINDOW)
    if available == 0:
        return 1
    # Roughly 64 candidate positions per channel, while retaining a small
    # stride for short series.  This scales with the amount of usable history.
    return max(1, int(np.ceil(available / 64)))


def candidates(rows, stride: int | None = None):
    result = []
    for channel, row in enumerate(rows):
        channel_stride = stride or adaptive_stride(len(row["values"]))
        for start in range(0, len(row["values"]) - WINDOW + 1, channel_stride):
            result.append((channel, start))
        last = len(row["values"]) - WINDOW
        if last >= 0 and (not result or result[-1] != (channel, last)):
            result.append((channel, last))
    return result


def select_diverse(cands, n_channels: int, n: int, rng: np.random.Generator):
    """Stratify first by channel and then by temporal quantile."""
    groups = {(c, q): [] for c in range(n_channels) for q in range(4)}
    by_channel = {c: [] for c in range(n_channels)}
    for idx, (c, _) in enumerate(cands):
        by_channel[c].append(idx)
    for c, indices in by_channel.items():
        for rank, idx in enumerate(indices):
            groups[(c, min(3, 4 * rank // max(1, len(indices))))].append(idx)

    # Round-robin over channels and temporal quartiles. This covers all
    # channels when possible and remains valid when n exceeds candidate count.
    buckets = [groups[(c, q)] for q in range(4) for c in range(n_channels)]
    for bucket in buckets:
        rng.shuffle(bucket)
    selected = []
    cursor = 0
    while len(selected) < min(n, len(cands)):
        progressed = False
        for bucket in buckets:
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) == min(n, len(cands)):
                    break
        if not progressed:
            break
        cursor += 1
    return [cands[i] for i in selected]


def describe(values: np.ndarray, domain: str, duration: float, unit: str) -> str:
    x = np.asarray(values, dtype=float)
    scale = float(np.std(x)) + 1e-12
    z = (x - np.mean(x)) / scale
    slope = float(np.polyfit(np.arange(len(x)), z, 1)[0] * len(x))
    means = np.array([np.mean(a) for a in np.array_split(x, 4)])
    stds = np.array([np.std(a) for a in np.array_split(x, 4)])
    rough = float(np.std(np.diff(x)) / scale)
    # Search far enough to detect the daily cycle in 15-minute data (96 points)
    # while retaining the short-cycle behavior of hourly/monthly series.
    ac = [] if scale <= 1e-10 else [
        float(np.corrcoef(z[:-lag], z[lag:])[0, 1])
        for lag in range(1, min(128, len(x) - 1))
    ]
    best_lag = 1 + int(np.nanargmax(np.nan_to_num(ac, nan=-1))) if ac else 0
    periodic = best_lag >= 2 and ac[best_lag - 1] > 0.55
    if slope > 0.35:
        trend = "an overall upward trend"
    elif slope < -0.35:
        trend = "an overall downward trend"
    else:
        trend = "no strong overall trend"
    changes = np.diff(means) / scale
    local = []
    if np.any(changes[:2] > 0.35):
        local.append("a local rise")
    if np.any(changes[:2] < -0.35):
        local.append("a local fall")
    if changes[-1] > 0.35:
        local.append("a late rise")
    elif changes[-1] < -0.35:
        local.append("a late fall")
    local_text = ", including " + " and ".join(local[:2]) if local else ""
    variation = "low" if rough < 0.12 else "moderate" if rough < 0.35 else "high"
    changing_vol = np.max(stds) > 1.5 * (np.mean(stds) + 1e-12)
    volatility = f"{variation} short-term variation"
    if changing_vol:
        volatility += " and changing volatility"
    cycle = f"a recurring pattern at lag about {best_lag}" if periodic else "no clear short-lag periodicity"
    return (
        f"A {domain} time-series window containing 128 consecutive observations "
        f"(about {duration:g} {unit}). The series shows {trend}{local_text}, "
        f"{volatility}, and {cycle}."
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_records = []
    configs = {}
    for name in DATASETS:
        rows, value_col = load_dataset(name)
        lengths = [len(row["values"]) for row in rows]
        length = min(lengths)
        strides = [adaptive_stride(item) for item in lengths]
        stride = int(np.median(strides))
        cands = candidates(rows)
        chosen = select_diverse(cands, len(rows), N_SAMPLES, rng)
        domain, unit = DOMAIN[name]
        step_hours = {"hours": 1, "months": 24 * 30, "days": 24}.get(unit, 1)
        duration = (WINDOW - 1) * step_hours / (24 if unit == "hours" else 1)
        # For hourly data express duration in hours; for monthly/daily use native units.
        if name == "electricity_15min":
            duration, duration_unit = WINDOW / 4, "hours"
        elif unit == "hours":
            duration, duration_unit = WINDOW, "hours"
        elif unit == "days":
            duration, duration_unit = WINDOW, "days"
        else:
            duration, duration_unit = WINDOW, "months"
        records = []
        fig, axes = plt.subplots(10, 10, figsize=(22, 20), constrained_layout=True)
        axes = axes.ravel()
        for sample_id, (channel, start) in enumerate(chosen):
            row = rows[channel]
            window = row["values"][start : start + WINDOW]
            timestamps = row["timestamp"][start : start + WINDOW]
            record = {
                "dataset": name, "sample_id": sample_id, "channel_id": row["id"],
                "channel_index": channel, "start_index": start, "window_length": len(window),
                "stride": stride, "start_time": str(timestamps[0]), "end_time": str(timestamps[-1]),
                "value_column": value_col, "mean": float(np.mean(window)), "std": float(np.std(window)),
                "min": float(np.min(window)), "max": float(np.max(window)),
                "caption": describe(window, domain, duration, duration_unit),
            }
            records.append(record); all_records.append(record)
            axes[sample_id].plot(window, lw=0.7); axes[sample_id].grid(alpha=0.2)
            axes[sample_id].set_title(f"{sample_id}: {row['id']} @ {start}", fontsize=7)
            axes[sample_id].tick_params(labelsize=6)
        for ax in axes[len(chosen):]: ax.axis("off")
        fig.suptitle(f"{name}: 100 channel-diverse contiguous windows (L=128, stride={stride})")
        fig.savefig(OUT / f"{name}_100_windows.png", dpi=160); fig.savefig(OUT / f"{name}_100_windows.pdf"); plt.close(fig)
        fields = list(records[0])
        with (OUT / f"{name}_samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)
        with (OUT / f"{name}_descriptions.jsonl").open("w") as f:
            for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        configs[name] = {"channels": len(rows), "min_length": min(lengths), "max_length": max(lengths), "strides": {"min": min(strides), "median": int(np.median(strides)), "max": max(strides)}, "candidates": len(cands), "selected": len(chosen), "value_column": value_col}
    (OUT / "sampling_config.json").write_text(json.dumps({"window": WINDOW, "seed": SEED, "datasets": configs}, indent=2))
    (OUT / "all_descriptions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records))
    print(json.dumps(configs, indent=2))


if __name__ == "__main__": main()