"""Shape-stratified window sampling from the 102 screened channels.

Problem: uniform-stride slicing over-samples the most common shape (monotone
slope / flat segments, e.g. MT_092@41496).  Fix: densely slice candidates,
characterize each window by (slope, curvature) on the z-scored window, split
into 3x3 shape bins by quantiles, then sample each bin equally so the final
~30,000 windows cover the full shape spectrum.

Outputs under results/trend_windows_ma65_screened_stratified/:
  - windows.npy / meta.csv (with shape_slope, shape_curv, shape_bin)
  - summary.json
  - sample_grid.png             : 11 windows per shape bin (99 total)
  - shape_distribution.png      : candidate vs selected (slope, curv) scatter
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
OUT = ROOT / "results" / "trend_windows_ma65_screened_stratified"

NAME = "electricity_15min"
COL = "consumption_kW"
WINDOW = 128
MA_WIN = 65
CAND_STRIDE = 32          # dense candidate slicing
TARGET_TOTAL = 30_000
N_BINS_SLOPE = 3          # down / flat / up
N_BINS_CURV = 3           # concave / flat / convex
SEED = 42

# precomputed design matrices for vectorized 1st/2nd order fits
X = np.arange(WINDOW, dtype=np.float64)
P1 = np.linalg.pinv(np.column_stack([X, np.ones(WINDOW)]))          # (2, 128)
P2 = np.linalg.pinv(np.column_stack([X ** 2, X, np.ones(WINDOW)]))  # (3, 128)


def smooth(s: np.ndarray, win: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def shape_features(W: np.ndarray):
    """Per-window (slope, curvature) on z-scored windows, vectorized."""
    z = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)
    slope = z @ P1.T[:, 0]   # linear coefficient
    curv = z @ P2.T[:, 0]    # quadratic coefficient
    return slope, curv


def main():
    out_dir = OUT / f"{NAME}_ma{MA_WIN}_w{WINDOW}_stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(SCREEN.read_text())
    kept = sorted({k for p in ["p1", "p2", "p3", "p4"]
                   for k in cfg["pages"][p]["kept_1based"]})
    kept_ids = [f"MT_{k:03d}" for k in kept]

    table = pa.ipc.open_stream(
        str(ARROW_ROOT / NAME / "train-00000-of-00001.arrow")).read_all()
    ids = table.column("id").to_pylist()
    idx = {cid: i for i, cid in enumerate(ids)}

    # ---- dense candidates ----
    cand_windows, cand_meta = [], []
    for ci, cid in enumerate(kept_ids):
        s = np.asarray(table.column(COL)[idx[cid]].as_py(), dtype=np.float64)
        z = smooth(s[np.isfinite(s)], MA_WIN)
        for st in range(0, len(z) - WINDOW + 1, CAND_STRIDE):
            cand_windows.append(z[st:st + WINDOW].astype(np.float32))
            cand_meta.append((cid, ci, st))
    W = np.stack(cand_windows)
    print(f"candidates: {len(W):,}")

    # ---- shape features + quantile bins ----
    slope, curv = shape_features(W)
    sq = np.quantile(slope, [1 / 3, 2 / 3])
    cq = np.quantile(curv, [1 / 3, 2 / 3])
    s_bin = np.digitize(slope, sq)   # 0/1/2
    c_bin = np.digitize(curv, cq)    # 0/1/2
    bin_id = s_bin * N_BINS_CURV + c_bin
    bin_names = []
    for sb in range(N_BINS_SLOPE):
        for cb in range(N_BINS_CURV):
            bin_names.append(f"s{sb}c{cb}")

    # ---- stratified sampling ----
    rng = np.random.default_rng(SEED)
    per_bin = TARGET_TOTAL // (N_BINS_SLOPE * N_BINS_CURV)
    picked = []
    for b in range(N_BINS_SLOPE * N_BINS_CURV):
        members = np.where(bin_id == b)[0]
        take = min(per_bin, len(members))
        picked.append(rng.choice(members, size=take, replace=False))
    picked = np.concatenate(picked)

    windows = W[picked]
    meta_rows = []
    for i in picked:
        cid, ci, st = cand_meta[i]
        w = windows[len(meta_rows)]
        meta_rows.append({
            "dataset": NAME, "channel_id": cid, "channel_index": ci,
            "window_index": -1, "start_index": int(st),
            "stride": CAND_STRIDE, "shape_slope": float(slope[i]),
            "shape_curv": float(curv[i]), "shape_bin": bin_names[bin_id[i]],
            "window_mean": float(np.mean(w)), "window_std": float(np.std(w)),
        })

    np.save(out_dir / "windows.npy", windows)
    with (out_dir / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        writer.writeheader()
        writer.writerows(meta_rows)

    summary = {
        "dataset": NAME, "ma_win": MA_WIN, "window": WINDOW,
        "cand_stride": CAND_STRIDE, "channels": len(kept_ids),
        "candidates": int(len(W)), "total_windows": int(len(windows)),
        "shape_bins": bin_names, "per_bin": per_bin,
        "slope_quantiles": [float(v) for v in sq],
        "curv_quantiles": [float(v) for v in cq],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- shape distribution: candidates vs selected ----
    rng2 = np.random.default_rng(SEED + 1)
    sub = rng2.choice(len(W), size=min(5000, len(W)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    axes[0].scatter(slope[sub], curv[sub], s=4, alpha=0.4)
    axes[0].set_title(f"candidates (sample {len(sub):,})")
    axes[1].scatter(slope[picked], curv[picked], s=4, alpha=0.5)
    axes[1].set_title(f"selected ({len(picked):,}, {N_BINS_SLOPE}x{N_BINS_CURV} bins)")
    for ax in axes:
        ax.set_xlabel("slope")
        ax.set_ylabel("curvature")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{NAME} — shape distribution: uniform-stride candidates vs "
                 f"stratified selection", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "shape_distribution.png", dpi=120)
    plt.close(fig)

    # ---- sample grid: 11 windows per shape bin ----
    fig, axes = plt.subplots(9, 11, figsize=(22, 16), constrained_layout=True)
    for b in range(9):
        members = np.where(bin_id[picked] == b)[0]
        show = members[:11]
        for c, i in enumerate(show):
            ax = axes[b, c]
            ax.plot(windows[i], lw=0.7)
            ax.set_title(f"{meta_rows[i]['channel_id']}@{meta_rows[i]['start_index']}",
                         fontsize=6)
            ax.tick_params(labelsize=5)
        axes[b, 0].set_ylabel(bin_names[b], fontsize=10)
        for c in range(len(show), 11):
            axes[b, c].axis("off")
    fig.suptitle(f"{NAME} — stratified windows, 11 per shape bin "
                 f"({len(windows):,} total)", fontsize=13)
    fig.savefig(out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
