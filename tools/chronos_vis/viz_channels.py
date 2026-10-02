#!/usr/bin/env python
"""Per-channel visualization for the 5 candidate chronos arrow datasets.

Each row in an arrow shard is one independent series (one channel, identified
by its `id`).  This script picks evenly-spaced channels per dataset and:
  - saves one full-series PNG per channel under figs/<dataset>/channels/<id>.png
  - saves one collage PNG per dataset under figs/<dataset>/<dataset>_channels.png
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

N_CHANNELS = 8          # channels picked per dataset (<= n)
MAX_PLOT_POINTS = 40_000  # cap for very long series (ercot ~155k)


def get_timestamps(table, i):
    """Return datetime64 array for row i, or None."""
    if "timestamp" not in table.column_names:
        return None
    try:
        ts = table.column("timestamp")[i].as_py()
        return np.asarray(ts, dtype="datetime64[ms]")
    except Exception:
        return None


def draw_channel(ax, table, col, i, ids):
    """Plot one channel on ax; return stats dict."""
    s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
    ts = get_timestamps(table, i)
    if len(s) > MAX_PLOT_POINTS:
        s, ts = s[:MAX_PLOT_POINTS], (ts[:MAX_PLOT_POINTS] if ts is not None else None)
    if ts is not None and len(ts) == len(s):
        ax.plot(ts, s, lw=0.7)
    else:
        ax.plot(s, lw=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7)
    return {
        "len": int(len(s)),
        "mean": float(np.nanmean(s)),
        "std": float(np.nanstd(s)),
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
    }


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
        chan_dir = os.path.join(OUT, name, "channels")
        os.makedirs(chan_dir, exist_ok=True)

        # per-channel full figure
        for i in idxs:
            cid = ids[i]
            fig, ax = plt.subplots(figsize=(14, 4))
            stats = draw_channel(ax, table, col, i, ids)
            ax.set_title(
                f"{name} / channel id={cid}   "
                f"(len={stats['len']:,}, mean={stats['mean']:.4g}, "
                f"std={stats['std']:.4g}, min={stats['min']:.4g}, "
                f"max={stats['max']:.4g})",
                fontsize=10,
            )
            fig.tight_layout()
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(cid))
            fig.savefig(os.path.join(chan_dir, f"{safe}.png"), dpi=100)
            plt.close(fig)

        # collage: one subplot per picked channel
        cols_n = 4 if n > 4 else n
        rows_n = int(np.ceil(n / cols_n))
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 4.2, rows_n * 2.6),
                                 squeeze=False)
        for k, i in enumerate(idxs):
            ax = axes[k // cols_n, k % cols_n]
            draw_channel(ax, table, col, i, ids)
            ax.set_title(f"id={ids[i]}", fontsize=9)
        for k in range(n, rows_n * cols_n):
            axes[k // cols_n, k % cols_n].axis("off")
        fig.suptitle(f"{name} — {n} picked channels (of {table.num_rows})",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_png = os.path.join(OUT, name, f"{name}_channels.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        print(f"{name}: {n} channels -> {chan_dir} + {out_png}")
        del table


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Per-channel visualization for the 5 candidate chronos arrow datasets.

Each row in an arrow shard is one independent series (one channel, identified
by its `id`).  This script picks evenly-spaced channels per dataset and:
  - saves one full-series PNG per channel under figs/<dataset>/channels/<id>.png
  - saves one collage PNG per dataset under figs/<dataset>/<dataset>_channels.png
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

N_CHANNELS = 8          # channels picked per dataset (<= n)
MAX_PLOT_POINTS = 40_000  # cap for very long series (ercot ~155k)


def get_timestamps(table, i):
    """Return datetime64 array for row i, or None."""
    if "timestamp" not in table.column_names:
        return None
    try:
        ts = table.column("timestamp")[i].as_py()
        return np.asarray(ts, dtype="datetime64[ms]")
    except Exception:
        return None


def draw_channel(ax, table, col, i, ids, max_points=None):
    """Plot one channel on ax; return stats dict."""
    s = np.asarray(table.column(col)[i].as_py(), dtype=np.float64)
    ts = get_timestamps(table, i)
    if max_points is not None and len(s) > max_points:
        s, ts = s[:max_points], (ts[:max_points] if ts is not None else None)
    if ts is not None and len(ts) == len(s):
        ax.plot(ts, s, lw=0.7)
    else:
        ax.plot(s, lw=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7)
    return {
        "len": int(len(s)),
        "mean": float(np.nanmean(s)),
        "std": float(np.nanstd(s)),
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
    }


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
        chan_dir = os.path.join(OUT, name, "channels")
        os.makedirs(chan_dir, exist_ok=True)

        # per-channel full figure
        for i in idxs:
            cid = ids[i]
            fig, ax = plt.subplots(figsize=(14, 4))
            stats = draw_channel(ax, table, col, i, ids)
            ax.set_title(
                f"{name} / channel id={cid}   "
                f"(len={stats['len']:,}, mean={stats['mean']:.4g}, "
                f"std={stats['std']:.4g}, min={stats['min']:.4g}, "
                f"max={stats['max']:.4g})",
                fontsize=10,
            )
            fig.tight_layout()
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(cid))
            fig.savefig(os.path.join(chan_dir, f"{safe}.png"), dpi=100)
            plt.close(fig)

        # collage: one subplot per picked channel
        cols_n = 4 if n > 4 else n
        rows_n = int(np.ceil(n / cols_n))
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 4.2, rows_n * 2.6),
                                 squeeze=False)
        for k, i in enumerate(idxs):
            ax = axes[k // cols_n, k % cols_n]
            draw_channel(ax, table, col, i, ids)
            ax.set_title(f"id={ids[i]}", fontsize=9)
        for k in range(n, rows_n * cols_n):
            axes[k // cols_n, k % cols_n].axis("off")
        fig.suptitle(f"{name} — {n} picked channels (of {table.num_rows})",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_png = os.path.join(OUT, name, f"{name}_channels.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        print(f"{name}: {n} channels -> {chan_dir} + {out_png}")
        del table


if __name__ == "__main__":
    main()
