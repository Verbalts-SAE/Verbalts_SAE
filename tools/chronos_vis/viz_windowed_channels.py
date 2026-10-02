#!/usr/bin/env python
"""MA-129 trend windows (win=128) visualization for the two accepted datasets.

A 129-point centered moving average removes all peaks/valleys at the
128-window scale, leaving a clean trend background.  The trend is then cut
into 128-point windows; 12 windows evenly spread across each channel are
shown per figure.

Per channel one figure under figs/<dataset>/windowed128_s64_ma129_trend/:
  - row 1      : trend overview (windowed regions highlighted)
  - rows 2-3   : 12 evenly-spread trend windows
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

FILES = ["electricity_15min", "monash_electricity_hourly"]
WIN = 128
STRIDE = 64
N_CHANNELS = 8
N_WINDOWS_SHOWN = 12
MA_WIN = 129


def smooth(s, win):
    """Centered moving average of the whole channel (edge-padded)."""
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def main():
    for name in FILES:
        path = os.path.join(BASE, name, "train-00000-of-00001.arrow")
        table = pa.ipc.open_stream(path).read_all()
        col = [f.name for f in table.schema
               if (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
               and (pa.types.is_floating(f.type.value_type)
                    or pa.types.is_integer(f.type.value_type))][0]
        ids = table.column("id").to_pylist() if "id" in table.column_names else \
            [f"{name[:4]}_{i:03d}" for i in range(table.num_rows)]

        n = min(N_CHANNELS, table.num_rows)
        idxs = np.linspace(0, table.num_rows - 1, n).astype(int)
        out_dir = os.path.join(OUT, name, f"windowed{WIN}_s{STRIDE}_ma{MA_WIN}_trend")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n== {name}: rows={table.num_rows}, col={col}, "
              f"win={WIN}, stride={STRIDE}, MA={MA_WIN}")
        for i in idxs:
            cid = ids[i]
            s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
            s = s[np.isfinite(s)]
            z = smooth(s, MA_WIN)
            n_win = max(0, (len(s) - WIN) // STRIDE + 1)
            print(f"  {cid:22s} len={len(s):7d}  windows={n_win:6d}")
            if n_win < N_WINDOWS_SHOWN:
                continue

            starts = np.linspace(0, len(s) - WIN, N_WINDOWS_SHOWN).astype(int)

            fig = plt.figure(figsize=(19, 7.6))
            gs = fig.add_gridspec(3, 6, hspace=0.75, height_ratios=[1.0, 1, 1])

            ax_full = fig.add_subplot(gs[0, :])
            ax_full.plot(z, lw=0.5, color="tab:blue")
            for st in starts:
                ax_full.axvspan(st, st + WIN, color="tab:orange", alpha=0.15)
            ax_full.set_title(f"{name} / {cid} — MA-{MA_WIN} trend overview "
                              f"(len={len(z):,}, windows={n_win:,})", fontsize=10)
            ax_full.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax_full.tick_params(labelsize=8)

            for k, st in enumerate(starts):
                ax = fig.add_subplot(gs[1 + k // 6, k % 6])
                ax.plot(z[st:st + WIN], lw=1.0, color="tab:blue")
                ax.set_title(f"trend win #{k} [{st}:{st + WIN}]", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            fig.suptitle(
                f"{name} / {cid} — MA-{MA_WIN} trend, {N_WINDOWS_SHOWN} evenly-"
                f"spread {WIN}-point windows (stride={STRIDE})", fontsize=13)
            safe = "".join(c if c.isalnum() or c in "._-" else "_"
                           for c in str(cid))
            fig.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=100)
            plt.close(fig)
        del table
        print(f"figures saved -> {out_dir}")


if __name__ == "__main__":
    main()
