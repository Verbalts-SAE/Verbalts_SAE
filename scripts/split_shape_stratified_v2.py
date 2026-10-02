"""Shape-stratified sampling v2: +10% noise + in-bin correlation dedup.

Pipeline:
  1. dense candidates (stride 32) from the 102 screened channels, MA-65;
  2. (slope, curvature) 3x3 quantile bins;
  3. sample per_bin per bin, add 10%*std Gaussian noise;
  4. greedy in-bin dedup (Pearson corr > 0.95 => duplicate);
  5. refill each bin from remaining candidates (noise + same corr check)
     back to per_bin.

Outputs under results/trend_windows_ma65_screened_stratified/
  electricity_15min_ma65_w128_stratified_v2/:
  - windows.npy / meta.csv (shape_bin, noise_level)
  - summary.json / sample_grid.png
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
CAND_STRIDE = 32
TARGET_TOTAL = 30_000
N_BINS_SLOPE = 3
N_BINS_CURV = 3
NOISE_LEVEL = 0.10
DEDUP_CORR_THR = 0.95
SEED = 42

X = np.arange(WINDOW, dtype=np.float64)
P1 = np.linalg.pinv(np.column_stack([X, np.ones(WINDOW)]))
P2 = np.linalg.pinv(np.column_stack([X ** 2, X, np.ones(WINDOW)]))


def smooth(s: np.ndarray, win: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def shape_features(W: np.ndarray):
    z = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)
    return z @ P1.T[:, 0], z @ P2.T[:, 0]


def zscore_rows(W: np.ndarray):
    return (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)


def greedy_dedup(Z: np.ndarray, rng: np.random.Generator, thr: float):
    """Greedy correlation-based dedup of z-scored rows; returns kept indices."""
    C = Z @ Z.T / Z.shape[1]
    order = rng.permutation(len(Z))
    keep = []
    for i in order:
        if keep and C[i, keep].max() >= thr:
            continue
        keep.append(i)
    return np.asarray(keep)


def main():
    out_dir = OUT / f"{NAME}_ma{MA_WIN}_w{WINDOW}_stratified_v2"
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
    n_cand = len(W)
    print(f"candidates: {n_cand:,}")

    slope, curv = shape_features(W)
    sq = np.quantile(slope, [1 / 3, 2 / 3])
    cq = np.quantile(curv, [1 / 3, 2 / 3])
    s_bin = np.digitize(slope, sq)
    c_bin = np.digitize(curv, cq)
    bin_id = s_bin * N_BINS_CURV + c_bin
    bin_names = [f"s{sb}c{cb}" for sb in range(N_BINS_SLOPE)
                 for cb in range(N_BINS_CURV)]
    n_bins = N_BINS_SLOPE * N_BINS_CURV
    per_bin = TARGET_TOTAL // n_bins

    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((n_cand, WINDOW)).astype(np.float32)
    stds = W.std(axis=1, keepdims=True)
    Wn_all = (W + NOISE_LEVEL * stds * noise).astype(np.float32)

    picked, bin_counts = [], []
    for b in range(n_bins):
        members = np.where(bin_id == b)[0]
        init = rng.choice(members, size=min(per_bin, len(members)), replace=False)
        Z = zscore_rows(Wn_all[init].astype(np.float64))
        keep = greedy_dedup(Z, rng, DEDUP_CORR_THR)
        kept_global = init[keep].tolist()

        # refill from remaining candidates
        rest = np.setdiff1d(members, init)
        rng.shuffle(rest)
        Zk = Z[keep].astype(np.float64)
        batch = 256
        for s0 in range(0, len(rest), batch):
            if len(kept_global) >= per_bin:
                break
            sub = rest[s0:s0 + batch]
            Zw = zscore_rows(Wn_all[sub].astype(np.float64))
            C = Zw @ Zk.T / WINDOW
            for j, i in enumerate(sub):
                if len(kept_global) >= per_bin:
                    break
                if C[j].max() < DEDUP_CORR_THR:
                    kept_global.append(i)
                    Zk = np.vstack([Zk, Zw[j]])
        picked.extend(kept_global)
        bin_counts.append(len(kept_global))
        print(f"bin {bin_names[b]:4s}: init={len(init)} -> dedup={len(keep)} "
              f"-> refilled={len(kept_global)}")

    picked = np.asarray(picked)
    windows = Wn_all[picked]
    meta_rows = []
    for i in picked:
        cid, ci, st = cand_meta[i]
        w = windows[len(meta_rows)]
        meta_rows.append({
            "dataset": NAME, "channel_id": cid, "channel_index": ci,
            "window_index": -1, "start_index": int(st), "stride": CAND_STRIDE,
            "shape_slope": float(slope[i]), "shape_curv": float(curv[i]),
            "shape_bin": bin_names[bin_id[i]], "noise_level": NOISE_LEVEL,
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
        "candidates": n_cand, "total_windows": int(len(windows)),
        "shape_bins": bin_names, "per_bin": per_bin,
        "noise_level": NOISE_LEVEL, "dedup_corr_thr": DEDUP_CORR_THR,
        "bin_counts": bin_counts, "seed": SEED,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- sample grid: 11 per bin ----
    fig, axes = plt.subplots(n_bins, 11, figsize=(22, 16), constrained_layout=True)
    for b in range(n_bins):
        members = np.where(np.asarray([m["shape_bin"] for m in meta_rows])
                           == bin_names[b])[0]
        show = members[:11]
        for c, i in enumerate(show):
            axes[b, c].plot(windows[i], lw=0.7)
            axes[b, c].tick_params(labelsize=5)
            axes[b, c].set_xticks([])
        axes[b, 0].set_ylabel(bin_names[b], fontsize=10)
        for c in range(len(show), 11):
            axes[b, c].axis("off")
    fig.suptitle(f"{NAME} — stratified + noise{NOISE_LEVEL} + "
                 f"dedup{DEDUP_CORR_THR} ({len(windows):,} windows)", fontsize=13)
    fig.savefig(out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
