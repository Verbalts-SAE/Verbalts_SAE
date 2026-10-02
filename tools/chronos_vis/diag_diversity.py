#!/usr/bin/env python
"""Compare window diversity across MA levels (129 / 65 / 31), uniform 81
windows per channel, for both accepted datasets.

Rows = MA levels, columns = 12 sampled windows.
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "diag")
os.makedirs(OUT, exist_ok=True)

WINDOW = 128
PER = 81
MA_LEVELS = [129, 65, 31]
DATASETS = [
    ("electricity_15min", "consumption_kW", [0, 159]),
    ("monash_electricity_hourly", "target", [0, 160]),
]


def smooth(s, win):
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def starts_uniform(length, per):
    avail = length - WINDOW
    stride = max(1, avail // (per - 1)) if per > 1 else 1
    return [wi * stride for wi in range(per)]


def main():
    for name, col, channels in DATASETS:
        table = pa.ipc.open_stream(
            os.path.join(BASE, name, "train-00000-of-00001.arrow")).read_all()
        ids = table.column("id").to_pylist()

        for ci in channels:
            cid = ids[ci]
            s = np.asarray(table.column(col)[ci].as_py(), dtype=np.float64)
            s = s[np.isfinite(s)]
            fig, axes = plt.subplots(len(MA_LEVELS), 12,
                                     figsize=(24, 3.2 * len(MA_LEVELS)),
                                     squeeze=False)
            for r, ma in enumerate(MA_LEVELS):
                z = smooth(s, ma)
                starts = starts_uniform(len(z), PER)
                show = np.linspace(0, PER - 1, 12).astype(int)
                for c, k in enumerate(show):
                    st = starts[k]
                    ax = axes[r, c]
                    ax.plot(z[st:st + WINDOW], lw=1.0)
                    ax.set_title(f"#{k} @{st}", fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.xaxis.set_visible(False)
                axes[r, 0].set_ylabel(f"MA-{ma}", fontsize=10)
            fig.suptitle(f"{name} / {cid} — MA level comparison, "
                         f"uniform {PER} windows/channel", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(os.path.join(OUT, f"ma_compare_{name}_{cid}.png"), dpi=110)
            plt.close(fig)
            print(f"saved ma_compare_{name}_{cid}.png")


if __name__ == "__main__":
    main()
