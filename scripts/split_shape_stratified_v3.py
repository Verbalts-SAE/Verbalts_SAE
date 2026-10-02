"""Threshold scan for correlation dedup (v3, with candidate cache).

Caches the noisy candidate pool (generated once by v2), then:
  A. scans dedup thresholds on the 9-bin init samples -> counts table;
  B. previews one threshold's sample grid.

Outputs under results/trend_windows_ma65_screened_stratified/_cache/ and
results/trend_windows_ma65_screened_stratified/..._v3_scan/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
OUT = ROOT / "results" / "trend_windows_ma65_screened_stratified"
CACHE = OUT / "_cache"
NAME = "electricity_15min"
WINDOW = 128
N_BINS_SLOPE = 3
N_BINS_CURV = 3
PER_BIN = 3333
NOISE_LEVEL = 0.10
THRESHOLDS = [0.95, 0.97, 0.98, 0.99, 0.995]
PREVIEW_THR = 0.98
SEED = 42


def zscore_rows(W):
    return (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)


def greedy_dedup(Z, rng, thr):
    C = Z @ Z.T / Z.shape[1]
    order = rng.permutation(len(Z))
    keep = []
    for i in order:
        if keep and C[i, keep].max() >= thr:
            continue
        keep.append(i)
    return np.asarray(keep)


def main():
    scan_dir = OUT / f"{NAME}_ma65_w128_v3_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    Wn_all = np.load(CACHE / "Wn_all.npy")
    bin_id = np.load(CACHE / "bin_id.npy")
    cids = np.load(CACHE / "cids.npy")
    sts = np.load(CACHE / "sts.npy")
    print(f"cached candidates: {len(Wn_all):,}")

    bin_names = [f"s{sb}c{cb}" for sb in range(N_BINS_SLOPE)
                 for cb in range(N_BINS_CURV)]
    n_bins = N_BINS_SLOPE * N_BINS_CURV
    rng = np.random.default_rng(SEED)

    # ---- A: threshold scan (init samples only, same seed as v2) ----
    table_rows = []
    for thr in THRESHOLDS:
        counts, totals = [], 0
        for b in range(n_bins):
            members = np.where(bin_id == b)[0]
            init = rng.choice(members, size=min(PER_BIN, len(members)),
                              replace=False)
            Z = zscore_rows(Wn_all[init].astype(np.float64))
            keep = greedy_dedup(Z, rng, thr)
            counts.append(len(keep))
            totals += len(keep)
        table_rows.append((thr, totals, counts))
        print(f"thr={thr:<6}: total={totals:>7,}  per-bin={counts}")

    with (scan_dir / "threshold_scan.json").open("w") as f:
        json.dump([{"thr": t, "total": int(tot), "per_bin": list(map(int, c))}
                   for t, tot, c in table_rows], f, indent=2)

    # ---- B: preview grid at PREVIEW_THR ----
    thr = PREVIEW_THR
    picked_all = []
    for b in range(n_bins):
        members = np.where(bin_id == b)[0]
        init = rng.choice(members, size=min(PER_BIN, len(members)), replace=False)
        Z = zscore_rows(Wn_all[init].astype(np.float64))
        keep = greedy_dedup(Z, rng, thr)
        picked_all.append(init[keep])

    fig, axes = plt.subplots(n_bins, 11, figsize=(22, 16), constrained_layout=True)
    for b in range(n_bins):
        picked = picked_all[b]
        show = picked[:11]
        for c, i in enumerate(show):
            axes[b, c].plot(Wn_all[i], lw=0.7)
            axes[b, c].tick_params(labelsize=5)
            axes[b, c].set_xticks([])
        axes[b, 0].set_ylabel(f"{bin_names[b]} (n={len(picked)})", fontsize=10)
        for c in range(len(show), 11):
            axes[b, c].axis("off")
    tot = sum(len(p) for p in picked_all)
    fig.suptitle(f"{NAME} — dedup corr>{thr} preview, first 11/box kept "
                 f"({tot:,} total after dedup, no refill)", fontsize=13)
    fig.savefig(scan_dir / f"preview_thr_{thr}.png", dpi=120)
    plt.close(fig)
    print(f"saved -> {scan_dir}")


if __name__ == "__main__":
    main()
