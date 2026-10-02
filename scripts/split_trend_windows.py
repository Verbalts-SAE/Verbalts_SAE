"""Split MA-129 trend windows (128 pts) from the two accepted chronos datasets.

Pipeline per dataset:
  1. each arrow row = one channel -> 129-point centered MA (edge-padded) to
     remove fine-grain peaks/valleys, leaving the trend background;
  2. per-channel adaptive stride = (len - 128) // (per_channel - 1) so that
     every channel contributes exactly `per_channel` evenly-spread windows;
  3. total windows ~= 30,000 per dataset (370 x 81 / 321 x 93).

Outputs under results/trend_windows_ma129/<dataset>/:
  - windows.npy   : [N, 128] float32
  - meta.csv      : dataset, channel_id, channel_index, window_index,
                    start_index, stride, window_mean, window_std
  - summary.json  : per-channel / stride statistics
  - sample_grid.png : 100 random windows for a visual sanity check
"""
from pathlib import Path

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
ARROW_ROOT = Path("/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow")
OUT = ROOT / "results" / "trend_windows_ma129"

WINDOW = 128
MA_WIN = 129
TARGET_TOTAL = 30_000
SEED = 42
DATASETS = ["electricity_15min", "monash_electricity_hourly"]


def smooth(s: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average of the whole channel (edge-padded)."""
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary_all = {}

    for name in DATASETS:
        path = ARROW_ROOT / name / "train-00000-of-00001.arrow"
        table = pa.ipc.open_stream(str(path)).read_all()
        col = [f.name for f in table.schema
               if (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
               and (pa.types.is_floating(f.type.value_type)
                    or pa.types.is_integer(f.type.value_type))][0]
        ids = table.column("id").to_pylist()
        n_ch = table.num_rows
        per = max(1, int(round(TARGET_TOTAL / n_ch)))  # windows per channel

        out_dir = OUT / f"{name}_ma{MA_WIN}_w{WINDOW}"
        out_dir.mkdir(parents=True, exist_ok=True)

        windows = []
        meta_rows = []
        strides = []
        for ci in range(n_ch):
            cid = ids[ci]
            s = np.asarray(table.column(col)[ci].as_py(), dtype=np.float64)
            s = s[np.isfinite(s)]
            z = smooth(s, MA_WIN)
            avail = len(z) - WINDOW
            stride = max(1, avail // (per - 1)) if per > 1 else 1
            strides.append(stride)
            for wi in range(per):
                st = wi * stride
                w = z[st:st + WINDOW].astype(np.float32)
                windows.append(w)
                meta_rows.append({
                    "dataset": name, "channel_id": cid, "channel_index": ci,
                    "window_index": wi, "start_index": int(st),
                    "stride": int(stride),
                    "window_mean": float(np.mean(w)),
                    "window_std": float(np.std(w)),
                })

        windows = np.stack(windows)  # [N, 128]
        np.save(out_dir / "windows.npy", windows)
        with (out_dir / "meta.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
            writer.writeheader()
            writer.writerows(meta_rows)

        strides = np.asarray(strides)
        summary = {
            "dataset": name,
            "value_column": col,
            "channels": n_ch,
            "per_channel": per,
            "total_windows": int(len(windows)),
            "window": WINDOW,
            "ma_win": MA_WIN,
            "stride": {"min": int(strides.min()), "median": int(np.median(strides)),
                       "max": int(strides.max())},
        }
        summary_all[name] = summary
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # visual sanity check: 100 random windows
        pick = rng.choice(len(windows), size=min(100, len(windows)), replace=False)
        fig, axes = plt.subplots(10, 10, figsize=(20, 16), constrained_layout=True)
        for ax, idx in zip(axes.ravel(), pick):
            ax.plot(windows[idx], lw=0.7)
            m = meta_rows[idx]
            ax.set_title(f"{m['channel_id']}@{m['start_index']}", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"{name} — {len(windows):,} MA-{MA_WIN} trend windows "
                     f"(100 random)", fontsize=13)
        fig.savefig(out_dir / "sample_grid.png", dpi=120)
        plt.close(fig)

        print(f"{name}: channels={n_ch}, per_channel={per}, "
              f"total={len(windows):,}, stride min/med/max="
              f"{strides.min()}/{int(np.median(strides))}/{strides.max()}")
        print(f"  -> {out_dir}")

    print(json.dumps(summary_all, indent=2))


if __name__ == "__main__":
    main()
