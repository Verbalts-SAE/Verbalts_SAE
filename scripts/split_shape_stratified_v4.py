"""Shape-stratified sampling v4: all 370 channels + stride 16 + noise + dedup.

Changes vs v2: use ALL channels (MT_001..MT_370, screened or not — window-level
corr dedup handles redundancy) and a denser candidate stride (16) to refill
closer to 30,000 windows while keeping the strict corr>0.95 dedup.

Pipeline:
  1. MA-65 smooth every channel, slice candidates at stride 16;
  2. (slope, curvature) 3x3 quantile bins;
  3. per bin: sample 3333, add 10%*std Gaussian noise;
  4. greedy in-bin dedup (Pearson corr > 0.95 => duplicate);
  5. refill each bin from remaining candidates (noise + same corr check).

Outputs under results/trend_windows_ma65_screened_stratified/
  electricity_15min_ma65_w128_stratified_v4/:
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
CAND_STRIDE = 16
TARGET_TOTAL = 30_000
N_BINS_SLOPE = 3
N_BINS_CURV = 3
NOISE_LEVEL = 0.10
DEDUP_CORR_THR = 0.95
SEED = 42
FEAT_CHUNK = 200_000   # rows per chunk for feature computation
REFILL_BATCH = 4096    # candidates per refill batch

X = np.arange(WINDOW, dtype=np.float64)
P1 = np.linalg.pinv(np.column_stack([X, np.ones(WINDOW)]))
P2 = np.linalg.pinv(np.column_stack([X ** 2, X, np.ones(WINDOW)]))


def smooth(s: np.ndarray, win: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, (pad, pad), mode="edge")
    return np.convolve(sp, k, mode="valid")


def zscore_rows(W: np.ndarray):
    return (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)


def greedy_dedup(Z: np.ndarray, rng: np.random.Generator, thr: float):
    C = Z @ Z.T / Z.shape[1]
    order = rng.permutation(len(Z))
    keep = []
    for i in order:
        if keep and C[i, keep].max() >= thr:
            continue
        keep.append(i)
    return np.asarray(keep)


def main():
    out_dir = OUT / f"{NAME}_ma{MA_WIN}_w{WINDOW}_stratified_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pa.ipc.open_stream(
        str(ARROW_ROOT / NAME / "train-00000-of-00001.arrow")).read_all()
    ids = table.column("id").to_pylist()
    idx = {cid: i for i, cid in enumerate(ids)}

    # channels: p1-p4 screened kept (102) + p5-p8 unscreened (MT_193..MT_370)
    cfg = json.loads(SCREEN.read_text())
    kept = sorted({k for p in ["p1", "p2", "p3", "p4"]
                   for k in cfg["pages"][p]["kept_1based"]})
    channel_ids = [f"MT_{k:03d}" for k in kept] + \
                  [f"MT_{k:03d}" for k in range(193, 371)]
    channel_ids = [c for c in sorted(set(channel_ids)) if c in idx]
    print(f"channels: {len(channel_ids)} (102 screened + p5-p8)")

    # ---- dense candidates (all channels, stride 16) ----
    cand_windows, cand_meta = [], []
    for ci, cid in enumerate(channel_ids):
        s = np.asarray(table.column(COL)[idx[cid]].as_py(), dtype=np.float64)
        z = smooth(s[np.isfinite(s)], MA_WIN)
        for st in range(0, len(z) - WINDOW + 1, CAND_STRIDE):
            cand_windows.append(z[st:st + WINDOW].astype(np.float32))
            cand_meta.append((cid, ci, st))
    W = np.stack(cand_windows)
    n_cand = len(W)
    print(f"candidates: {n_cand:,}")

    # ---- shape features (chunked) + quantile bins ----
    slope = np.empty(n_cand, dtype=np.float32)
    curv = np.empty(n_cand, dtype=np.float32)
    for s0 in range(0, n_cand, FEAT_CHUNK):
        z = zscore_rows(W[s0:s0 + FEAT_CHUNK].astype(np.float64))
        slope[s0:s0 + FEAT_CHUNK] = z @ P1.T[:, 0]
        curv[s0:s0 + FEAT_CHUNK] = z @ P2.T[:, 0]
    sq = np.quantile(slope, [1 / 3, 2 / 3])
    cq = np.quantile(curv, [1 / 3, 2 / 3])
    bin_id = np.digitize(slope, sq) * N_BINS_CURV + np.digitize(curv, cq)
    bin_names = [f"s{sb}c{cb}" for sb in range(N_BINS_SLOPE)
                 for cb in range(N_BINS_CURV)]
    n_bins = N_BINS_SLOPE * N_BINS_CURV
    per_bin = TARGET_TOTAL // n_bins

    # ---- noise (10% * window std, fixed seed) ----
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal(W.shape, dtype=np.float32)
    Wn_all = (W + NOISE_LEVEL * W.std(axis=1, keepdims=True) * noise).astype(np.float32)
    del W, noise, cand_windows

    # ---- per-bin: sample, dedup, refill ----
    picked, bin_counts = [], []
    for b in range(n_bins):
        members = np.where(bin_id == b)[0]
        init = rng.choice(members, size=min(per_bin, len(members)), replace=False)
        Z = zscore_rows(Wn_all[init].astype(np.float64))
        keep = greedy_dedup(Z, rng, DEDUP_CORR_THR)
        kept_global = init[keep].tolist()
        Zk = Z[keep].astype(np.float64)

        rest = np.setdiff1d(members, init)
        rng.shuffle(rest)
        for s0 in range(0, len(rest), REFILL_BATCH):
            if len(kept_global) >= per_bin:
                break
            sub = rest[s0:s0 + REFILL_BATCH]
            Zw = zscore_rows(Wn_all[sub].astype(np.float64))
            Cb = Zw @ Zk.T / WINDOW
            pass_j = np.where(Cb.max(axis=1) < DEDUP_CORR_THR)[0]
            if len(pass_j):
                # light in-batch dedup among survivors, then append at once
                Zc = Zw[pass_j]
                if len(Zc) > 1:
                    Cc = Zc @ Zc.T / WINDOW
                    order = rng.permutation(len(Zc))
                    keep2 = []
                    for j in order:
                        if keep2 and Cc[j, keep2].max() >= DEDUP_CORR_THR:
                            continue
                        keep2.append(j)
                    pass_j = pass_j[np.asarray(keep2)]
                kept_global.extend(int(i) for i in sub[pass_j])
                Zk = np.vstack([Zk, Zw[pass_j]])
                if len(kept_global) >= per_bin:
                    kept_global = kept_global[:per_bin]
                    break
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
        "cand_stride": CAND_STRIDE, "channels": len(channel_ids),
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
    fig.suptitle(f"{NAME} — v4: all channels, stride{CAND_STRIDE}, "
                 f"noise{NOISE_LEVEL}, dedup{DEDUP_CORR_THR} "
                 f"({len(windows):,} windows)", fontsize=13)
    fig.savefig(out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
