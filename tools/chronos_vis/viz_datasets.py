#!/usr/bin/env python
"""Enhanced visualization for the 5 candidate chronos arrow datasets.

Per dataset: 6 evenly-sampled series (real timestamps on x-axis when available)
plus series-length histogram and log-scale value histogram.
Plus one overview figure comparing a representative series from each dataset.
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "monash_electricity_hourly",
    "monash_fred_md",
    "exchange_rate",
    "ercot",
    "electricity_15min",
]

MAX_POINTS = 2500  # points per drawn series
N_SAMPLES = 6


def get_timestamps(table, i):
    """Return datetime64 array for row i, or None."""
    if "timestamp" not in table.column_names:
        return None
    try:
        ts = table.column("timestamp")[i].as_py()
        arr = np.asarray(ts, dtype="datetime64[ms]")
        return arr
    except Exception:
        return None


def main():
    overview = {}
    for name in FILES:
        path = os.path.join(BASE, name, "train-00000-of-00001.arrow")
        table = pa.ipc.open_stream(path).read_all()
        col = [f.name for f in table.schema
               if (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
               and (pa.types.is_floating(f.type.value_type)
                    or pa.types.is_integer(f.type.value_type))][0]

        lengths = []
        vals_all = []
        ids = table.column("id").to_pylist() if "id" in table.column_names else None
        for i in range(table.num_rows):
            s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
            lengths.append(len(s))
            step = max(1, len(s) // 100_000)
            vals_all.append(s[::step])
        vals_all = np.concatenate(vals_all)
        vals_all = vals_all[np.isfinite(vals_all)]
        lengths = np.asarray(lengths)

        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)

        idxs = np.linspace(0, table.num_rows - 1, N_SAMPLES).astype(int)
        for k, i in enumerate(idxs):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
            ts = get_timestamps(table, i)
            if len(s) > MAX_POINTS:
                s, ts = s[:MAX_POINTS], (ts[:MAX_POINTS] if ts is not None else None)
            if ts is not None and len(ts) == len(s):
                ax.plot(ts, s, lw=0.8)
            else:
                ax.plot(s, lw=0.8)
            label = f"row {i}" + (f"  id={ids[i]}" if ids else "")
            ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax_len = fig.add_subplot(gs[2, 0])
        ax_len.hist(lengths, bins=30, color="tab:blue", alpha=0.85)
        ax_len.set_title("series length", fontsize=10)
        ax_len.tick_params(labelsize=8)
        ax_len.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        ax_val = fig.add_subplot(gs[2, 1])
        lo, hi = np.percentile(vals_all, 0.5), np.percentile(vals_all, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = vals_all.min(), vals_all.max()
        ax_val.hist(vals_all, bins=100, range=(lo, hi), color="tab:orange", alpha=0.85)
        ax_val.set_yscale("log")
        ax_val.set_title("value dist (log y, 0.5-99.5 pct)", fontsize=10)
        ax_val.tick_params(labelsize=8)
        ax_val.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        ax_std = fig.add_subplot(gs[2, 2])
        stds = []
        for i in range(table.num_rows):
            s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
            stds.append(float(np.nanstd(s)))
        ax_std.hist(np.asarray(stds), bins=30, color="tab:green", alpha=0.85)
        ax_std.set_title("per-series std", fontsize=10)
        ax_std.tick_params(labelsize=8)

        fig.suptitle(
            f"{name}  —  n={table.num_rows} series, length "
            f"{lengths.min():,}~{lengths.max():,} (med {int(np.median(lengths)):,}), "
            f"values {vals_all.min():.4g}~{vals_all.max():.4g}, col={col}",
            fontsize=13,
        )
        out_png = os.path.join(OUT, f"{name}.png")
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_png}")

        # keep one representative normalized series for the overview
        rep = np.asarray(table.column(col)[0].as_py(), dtype=np.float64)
        rep = rep[np.isfinite(rep)]
        if rep.size > 2000:
            rep = rep[:2000]
        r = rep - rep.mean()
        overview[name] = r / (np.abs(r).max() + 1e-12)
        del table

    # ---- overview ----
    fig, axes = plt.subplots(len(FILES), 1, figsize=(14, 10), sharex=False)
    for ax, (name, y) in zip(axes, overview.items()):
        ax.plot(y, lw=0.9, color="tab:blue")
        ax.set_ylabel(name.replace("_", "\n"), fontsize=9, rotation=0,
                      labelpad=58, va="center")
        ax.tick_params(labelsize=8)
        ax.set_ylim(-1.15, 1.15)
    axes[-1].set_xlabel("time step (first 2000 points, standardized)", fontsize=10)
    fig.suptitle("Overview: one representative series per dataset (standardized)",
                 fontsize=13)
    out_png = os.path.join(OUT, "overview.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
