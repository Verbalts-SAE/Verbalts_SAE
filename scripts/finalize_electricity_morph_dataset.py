"""Finalize the electricity_15min morphology dataset for downstream tasks.

Input : results/trend_windows_ma65_screened/electricity_15min_ma65_w128_screened/
        (102 screened channels, MA-65, 29,988 windows)
Output: datasets/electricity_15min_morph/
        windows.npy (10%*std Gaussian noise added, seed 42)
        meta.csv (mean/std recomputed on noisy windows, noise_level column)
        summary.json / sample_grid.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
SRC = ROOT / "results/trend_windows_ma65_screened/electricity_15min_ma65_w128_screened"
DST = ROOT / "datasets/electricity_15min_morph"
NOISE_LEVEL = 0.10
SEED = 42


def main():
    DST.mkdir(parents=True, exist_ok=True)

    W = np.load(SRC / "windows.npy").astype(np.float32)
    with (SRC / "meta.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal(W.shape, dtype=np.float32)
    stds = W.std(axis=1, keepdims=True)
    Wn = (W + NOISE_LEVEL * stds * noise).astype(np.float32)

    for r, w in zip(rows, Wn):
        r["noise_level"] = NOISE_LEVEL
        r["window_mean"] = float(np.mean(w))
        r["window_std"] = float(np.std(w))

    np.save(DST / "windows.npy", Wn)
    with (DST / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    chans = sorted({r["channel_id"] for r in rows})
    strides = sorted({int(r["stride"]) for r in rows})
    summary = {
        "name": "electricity_15min_morph",
        "source_dataset": "chronos electricity_15min (train split)",
        "smoothing": "centered moving average, window 65, edge-padded",
        "window_length": 128,
        "channels": len(chans),
        "channel_ids": chans,
        "windows_per_channel": 294,
        "stride_range": [min(strides), max(strides)],
        "total_windows": int(len(Wn)),
        "noise": {"type": "gaussian", "level": f"{NOISE_LEVEL} * per-window std",
                  "seed": SEED},
        "dtype": "float32",
        "units": "kW (original amplitude, not normalized)",
        "files": ["windows.npy", "meta.csv", "summary.json",
                  "sample_grid.png", "DATASET_CARD.md"],
    }
    (DST / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    rng2 = np.random.default_rng(SEED + 1)
    pick = rng2.choice(len(Wn), size=min(100, len(Wn)), replace=False)
    fig, axes = plt.subplots(10, 10, figsize=(20, 16), constrained_layout=True)
    for ax, i in zip(axes.ravel(), pick):
        ax.plot(Wn[i], lw=0.7)
        r = rows[i]
        ax.set_title(f"{r['channel_id']}@{r['start_index']}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"electricity_15min_morph — {len(Wn):,} windows "
                 f"(MA-65, 102 channels, +{NOISE_LEVEL}*std noise)", fontsize=13)
    fig.savefig(DST / "sample_grid.png", dpi=120)
    plt.close(fig)
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
