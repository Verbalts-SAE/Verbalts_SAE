"""Split MA-65 trend windows from the 102 screened electricity_15min channels.

Pipeline:
  1. load the manually screened channel list (channel_screen_electricity_15min.json);
  2. per channel: 65-point centered MA (edge-padded) to keep the trend
     background while preserving some local shape;
  3. per-channel adaptive stride = (len - 128) // (per - 1), per ~= 294
     so the total is ~30,000 windows and every channel contributes equally.

Outputs under results/trend_windows_ma65_screened/:
  - windows.npy   : [N, 128] float32
  - meta.csv      : dataset, channel_id, channel_index, window_index,
                    start_index, stride, window_mean, window_std
  - summary.json
  - sample_grid.png           : 100 random windows
  - per_channel_overview_p*.png : 4-window overlay per screened channel
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
ARROW_ROOT = Path("/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow")
SCREEN = ROOT / "tools/chronos_vis/channel_screen_electricity_15min.json"
OUT = ROOT / "results" / "trend_windows_ma65_screened"

NAME = "electricity_15min"
COL = "consumption_kW"
WINDOW = 128
MA_WIN = 65
TARGET_TOTAL = 30_000
SEED = 42


def smooth(s: np.ndarray, win: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def main():
    out_dir = OUT / f"{NAME}_ma{MA_WIN}_w{WINDOW}_screened"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(SCREEN.read_text())
    kept = sorted({k for p in ["p1", "p2", "p3", "p4"]
                   for k in cfg["pages"][p]["kept_1based"]})
    kept_ids = [f"MT_{k:03d}" for k in kept]

    table = pa.ipc.open_stream(
        str(ARROW_ROOT / NAME / "train-00000-of-00001.arrow")).read_all()
    ids = table.column("id").to_pylist()
    idx = {cid: i for i, cid in enumerate(ids)}

    per = max(1, int(round(TARGET_TOTAL / len(kept_ids))))  # ~294

    windows, meta_rows, strides = [], [], []
    for ci, cid in enumerate(kept_ids):
        s = np.asarray(table.column(COL)[idx[cid]].as_py(), dtype=np.float64)
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
                "dataset": NAME, "channel_id": cid, "channel_index": ci,
                "window_index": wi, "start_index": int(st), "stride": int(stride),
                "window_mean": float(np.mean(w)), "window_std": float(np.std(w)),
            })

    windows = np.stack(windows)
    np.save(out_dir / "windows.npy", windows)
    with (out_dir / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        writer.writeheader()
        writer.writerows(meta_rows)

    strides = np.asarray(strides)
    summary = {
        "dataset": NAME, "value_column": COL, "ma_win": MA_WIN, "window": WINDOW,
        "channels": len(kept_ids), "per_channel": per,
        "total_windows": int(len(windows)),
        "stride": {"min": int(strides.min()), "median": int(np.median(strides)),
                   "max": int(strides.max())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- 100 random windows grid ----
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(windows), size=min(100, len(windows)), replace=False)
    fig, axes = plt.subplots(10, 10, figsize=(20, 16), constrained_layout=True)
    for ax, i in zip(axes.ravel(), pick):
        ax.plot(windows[i], lw=0.7)
        m = meta_rows[i]
        ax.set_title(f"{m['channel_id']}@{m['start_index']}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{NAME} — {len(windows):,} MA-{MA_WIN} windows from "
                 f"{len(kept_ids)} screened channels (100 random)", fontsize=13)
    fig.savefig(out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    # ---- per-channel 4-window overlay (screened channels only) ----
    per_page = 48
    n_pages = int(np.ceil(len(kept_ids) / per_page))
    for page in range(n_pages):
        fig, axes = plt.subplots(6, 8, figsize=(20.8, 13.2), squeeze=False)
        for slot, ax in enumerate(axes.ravel()):
            k = page * per_page + slot
            if k >= len(kept_ids):
                ax.axis("off")
                continue
            s = np.asarray(table.column(COL)[idx[kept_ids[k]]].as_py(),
                           dtype=np.float64)
            z = smooth(s[np.isfinite(s)], MA_WIN)
            starts = np.linspace(0, len(z) - WINDOW, 4).astype(int)
            for st in starts:
                ax.plot(z[st:st + WINDOW], lw=0.8)
            ax.set_title(f"{kept_ids[k]}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xticks([])
        fig.suptitle(f"{NAME} — screened channels ({page * per_page + 1}-"
                     f"{min((page + 1) * per_page, len(kept_ids))} of "
                     f"{len(kept_ids)}), MA-{MA_WIN}, 4 windows/channel",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"per_channel_overview_p{page + 1}.png", dpi=110)
        plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
