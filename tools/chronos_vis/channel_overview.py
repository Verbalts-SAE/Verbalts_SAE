#!/usr/bin/env python
"""All-channel MA-65 window-overlay collage for visual channel screening.

Each cell = one channel: 4 evenly-spread MA-65 windows overlaid.
Overlapping windows => monotonous channel (bad diversity);
spread windows => diverse shapes (good).
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "diag",
                   "channel_overview")
os.makedirs(OUT, exist_ok=True)

WINDOW = 128
MA = 65
N_WIN_PER_CHANNEL = 4
COLS, ROWS = 8, 6  # 48 channels per page
DATASETS = [
    ("electricity_15min", "consumption_kW"),
    ("monash_electricity_hourly", "target"),
]


def smooth(s, win):
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def main():
    for name, col in DATASETS:
        table = pa.ipc.open_stream(
            os.path.join(BASE, name, "train-00000-of-00001.arrow")).read_all()
        ids = table.column("id").to_pylist()
        n_ch = table.num_rows
        per_page = COLS * ROWS
        n_pages = int(np.ceil(n_ch / per_page))

        for page in range(n_pages):
            fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 2.6, ROWS * 2.2),
                                     squeeze=False)
            for slot, ax in enumerate(axes.ravel()):
                ci = page * per_page + slot
                if ci >= n_ch:
                    ax.axis("off")
                    continue
                s = np.asarray(table.column(col)[ci].as_py(), dtype=np.float64)
                s = s[np.isfinite(s)]
                z = smooth(s, MA)
                starts = np.linspace(0, len(z) - WINDOW,
                                     N_WIN_PER_CHANNEL).astype(int)
                for st in starts:
                    ax.plot(z[st:st + WINDOW], lw=0.8)
                ax.set_title(f"{ci}:{ids[ci]}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.set_xticks([])
            fig.suptitle(f"{name} — MA-{MA}, {N_WIN_PER_CHANNEL} windows "
                         f"overlaid per channel (page {page + 1}/{n_pages})",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = os.path.join(OUT, f"{name}_p{page + 1}.png")
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
