#!/usr/bin/env python
"""Probe & visualize candidate chronos arrow datasets.

For each dataset: print schema/size, compute per-series statistics and
global value distribution, then render a summary figure
(sampled series + series-length histogram + value histogram).
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

MAX_PLOT_POINTS = 1500  # per series when drawing
VAL_SAMPLE_PER_SERIES = 200_000  # points per series fed into global value histogram


def numeric_seq_columns(schema):
    """Return {name: field} for list-typed numeric sequence columns."""
    cols = {}
    for f in schema:
        if pa.types.is_list(f.type) or pa.types.is_large_list(f.type):
            vt = f.type.value_type
            if pa.types.is_floating(vt) or pa.types.is_integer(vt):
                cols[f.name] = f
    return cols


def main():
    report = []
    for name in FILES:
        path = os.path.join(BASE, name, "train-00000-of-00001.arrow")
        table = pa.ipc.open_stream(path).read_all()
        seq_cols = numeric_seq_columns(table.schema)

        print("=" * 90)
        print(f"dataset = {name}   rows = {table.num_rows}   "
              f"file = {os.path.getsize(path) / 1e6:.1f} MB")
        print("schema:")
        for f in table.schema:
            print(f"   - {f.name}: {f.type}")
        print(f"numeric sequence columns: {list(seq_cols.keys())}")

        # ---- per-series statistics over the first numeric sequence column ----
        if not seq_cols:
            print("!! no numeric sequence column, skipped")
            continue
        val_col = list(seq_cols.keys())[0]

        lengths, means, stds, mins, maxs = [], [], [], [], []
        ids = table.column("id").to_pylist() if "id" in table.column_names else None
        vals_all = []
        for i in range(table.num_rows):
            s = np.asarray(table.column(val_col)[i].as_py(), dtype=np.float64)
            lengths.append(len(s))
            means.append(float(np.nanmean(s)))
            stds.append(float(np.nanstd(s)))
            mins.append(float(np.nanmin(s)))
            maxs.append(float(np.nanmax(s)))
            step = max(1, len(s) // VAL_SAMPLE_PER_SERIES)
            vals_all.append(s[::step])
        vals_all = np.concatenate(vals_all)
        n_nan = int(np.isnan(vals_all).sum())
        vals_all = vals_all[np.isfinite(vals_all)]

        lengths = np.asarray(lengths)
        print(f"series stats (col={val_col}): n={table.num_rows}")
        print(f"  length : min={lengths.min()} median={np.median(lengths):.0f} "
              f"mean={lengths.mean():.1f} max={lengths.max()}")
        print(f"  mean   : min={np.min(means):.4g} max={np.max(means):.4g} "
              f"(overall {vals_all.mean():.4g})")
        print(f"  std    : min={np.min(stds):.4g} median={np.median(stds):.4g} "
              f"max={np.max(stds):.4g}")
        print(f"  value  : min={vals_all.min():.4g} p1={np.percentile(vals_all,1):.4g} "
              f"p99={np.percentile(vals_all,99):.4g} max={vals_all.max():.4g} "
              f"(nan_dropped={n_nan})")
        if ids:
            uniq = len(set(ids))
            print(f"  ids    : unique={uniq} (example: {ids[:4]})")

        # ---- figure ----
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])]

        # sample up to 4 series spread across the dataset
        idxs = np.linspace(0, table.num_rows - 1, min(4, table.num_rows)).astype(int)
        for ax, i in zip(axs, idxs):
            s = np.asarray(table.column(val_col)[i].as_py(), dtype=np.float64)
            if len(s) > MAX_PLOT_POINTS:
                s = s[:MAX_PLOT_POINTS]
            ax.plot(s, lw=0.8)
            label = f"row {i}" + (f" id={ids[i]}" if ids else "")
            ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=8)

        ax_len = fig.add_subplot(gs[1, 1])
        ax_len.hist(lengths, bins=30, color="tab:blue", alpha=0.8)
        ax_len.set_title("series length histogram", fontsize=10)
        ax_len.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax_len.tick_params(labelsize=8)

        ax_val = fig.add_subplot(gs[1, 2])
        lo, hi = np.percentile(vals_all, 0.5), np.percentile(vals_all, 99.5)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = vals_all.min(), vals_all.max()
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = 0.0, 1.0
        ax_val.hist(vals_all, bins=100, range=(lo, hi), color="tab:orange", alpha=0.8)
        ax_val.set_title("value histogram (0.5-99.5 pct)", fontsize=10)
        ax_val.tick_params(labelsize=8)

        fig.suptitle(f"{name}  (n={table.num_rows}, col={val_col})", fontsize=13)
        out_png = os.path.join(OUT, f"{name}.png")
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"figure saved -> {out_png}")

        report.append((name, table.num_rows, int(lengths.min()),
                       int(np.median(lengths)), int(lengths.max()), val_col))
        del table, vals_all

    print("\n" + "=" * 90)
    print("SUMMARY")
    for r in report:
        print(f"  {r[0]:28s} rows={r[1]:5d}  len min/med/max = "
              f"{r[2]}/{r[3]}/{r[4]}  col={r[5]}")


if __name__ == "__main__":
    main()
