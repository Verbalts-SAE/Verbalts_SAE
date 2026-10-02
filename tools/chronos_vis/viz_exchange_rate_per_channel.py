#!/usr/bin/env python
"""Per-channel visualization of the exchange_rate chronos arrow dataset.

Each row = one channel (currency_N, daily, 1990-01-01 .. 2019-01-30).
Outputs under figs/exchange_rate/per_channel/:
  - currency_N.png          : one full-series figure per channel (real dates)
  - overlay_normalized.png  : all 8 channels overlaid after z-score normalization
  - window_crisis.png       : 2007-2010 window (GFC) for 3 representative channels
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow"
PATH = os.path.join(BASE, "exchange_rate", "train-00000-of-00001.arrow")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figs", "exchange_rate", "per_channel")
os.makedirs(OUT, exist_ok=True)

CMAP = plt.get_cmap("tab10")


def main():
    table = pa.ipc.open_stream(PATH).read_all()
    ids = table.column("id").to_pylist()
    ts0 = np.asarray(table.column("timestamp")[0].as_py(), dtype="datetime64[ms]")

    series = {}
    for i in range(table.num_rows):
        s = np.asarray(table.column("target")[i].as_py(), dtype=np.float64)
        series[ids[i]] = s[np.isfinite(s)]

    # ---- 1) one full figure per channel ----
    for cid, s in series.items():
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts0, s, lw=0.8)
        ax.set_title(
            f"exchange_rate / {cid}  (len={len(s):,}, mean={s.mean():.4g}, "
            f"std={s.std():.4g}, min={s.min():.4g}, max={s.max():.4g})",
            fontsize=11)
        ax.set_ylabel("rate")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{cid}.png"), dpi=110)
        plt.close(fig)

    # ---- 2) z-score normalized overlay ----
    fig, ax = plt.subplots(figsize=(14, 5))
    for k, (cid, s) in enumerate(series.items()):
        z = (s - s.mean()) / s.std()
        ax.plot(ts0, z, lw=0.8, label=f"{cid} (raw std={s.std():.4g})",
                color=CMAP(k))
    ax.set_title("exchange_rate — all 8 channels, z-score normalized overlay")
    ax.set_ylabel("z-score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overlay_normalized.png"), dpi=110)
    plt.close(fig)

    # ---- 3) GFC window (2007-2010) for 3 representative channels ----
    t0 = np.datetime64("2007-01-01")
    t1 = np.datetime64("2010-01-01")
    m = (ts0 >= t0) & (ts0 < t1)
    pick = ["currency_1", "currency_2", "currency_5", "currency_6"]
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 3.2 * len(pick)),
                             sharex=True)
    for ax, cid in zip(axes, pick):
        s = series[cid]
        ax.plot(ts0[m], s[m], lw=1.0, color=CMAP(list(series).index(cid)))
        ax.set_title(f"{cid} — 2007..2010 window (GFC)", fontsize=10)
        ax.set_ylabel("rate")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle("exchange_rate — 2007-2010 detail for representative channels",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "window_crisis.png"), dpi=110)
    plt.close(fig)

    print(f"saved {len(series)} per-channel figures + overlay + window -> {OUT}")
    for cid, s in series.items():
        print(f"  {cid:11s} len={len(s):5d} mean={s.mean():.4g} std={s.std():.4g} "
              f"min={s.min():.4g} max={s.max():.4g}")


if __name__ == "__main__":
    main()
