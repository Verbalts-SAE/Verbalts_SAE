"""Visualize steering cases for etth2_2class verbalts_v2 (seed1).

Picks representative test cases from the full-test steering runs and plots
GT vs pure vs applied steering at increasing strengths:
  - rescued: pure wrong (3-stage CNN mismatch) -> strong applied correct
  - broken:  pure correct -> strongest applied wrong
  - mse_blowup: largest per-sample MSE vs GT under the strongest variant

Usage: python scripts/plot_etth2_steering_cases.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/etth2_2class/etth2_2class_dataset"
SEED1 = ROOT / "results/etth2_2class/seed1/verbalts_v2/steering"
OUT = ROOT / "outputs/etth2_steering_cases"
OUT.mkdir(parents=True, exist_ok=True)

# (dir, variant-name) pairs in increasing strength; curves + predictions.
RUNS = [
    (SEED1 / "boost2_fulltest_1023362", [("applied_c2_s1", "c2"),
                                          ("applied_c12_s4", "c12"),
                                          ("applied_c16_s5", "c16"),
                                          ("applied_c20_s6", "c20")]),
    (SEED1 / "boost3_fulltest_1023520", [("applied_c28_s8", "c28")]),
]

ts = np.load(DATA / "test_ts.npy")[:, :, 0].astype(np.float64)
labels = np.load(DATA / "test_labels.npy")  # (N,3) 0=single 1=double


def peaks_of(run_dir: Path, name: str) -> np.ndarray:
    pred = json.load(open(run_dir / f"{name}_predictions.json"))
    return np.array([[p["peak_count"] for p in row] for row in pred])  # (N,3)


def correct(peaks: np.ndarray) -> np.ndarray:
    return np.all(peaks == labels + 1, axis=1)


def load_curve(run_dir: Path, name: str) -> np.ndarray:
    return np.load(run_dir / f"{name}.npy").astype(np.float64)


pure_peaks = peaks_of(SEED1 / "boost3_fulltest_1023520", "pure")
pure_curves = load_curve(SEED1 / "boost3_fulltest_1023520", "pure")
c28_peaks = peaks_of(SEED1 / "boost3_fulltest_1023520", "applied_c28_s8")
c28_curves = load_curve(SEED1 / "boost3_fulltest_1023520", "applied_c28_s8")

pure_ok = correct(pure_peaks)
c28_ok = correct(c28_peaks)

rescued = np.where(~pure_ok & c28_ok)[0]
broken = np.where(pure_ok & ~c28_ok)[0]
mse_per_sample = ((c28_curves - ts) ** 2).mean(axis=1)
blowup = np.argsort(-mse_per_sample)

rng = np.random.default_rng(0)
rescued_pick = rng.choice(rescued, size=min(4, len(rescued)), replace=False)
broken_pick = rng.choice(broken, size=min(4, len(broken)), replace=False)
blowup_pick = blowup[:4]

print(f"rescued n={len(rescued)} picked={rescued_pick.tolist()}")
print(f"broken  n={len(broken)} picked={broken_pick.tolist()}")
print(f"blowup  picked={blowup_pick.tolist()} mse={mse_per_sample[blowup_pick].round(1).tolist()}")


def plot_case(ax, idx: int):
    t = np.arange(128)
    ax.plot(t, ts[idx], "k--", lw=1.6, label="GT", zorder=5)
    ax.plot(t, pure_curves[idx], color="#4477cc", lw=1.4, label="pure", zorder=4)
    cmap = plt.cm.YlOrRd
    i = 0
    for run_dir, items in RUNS:
        for fname, short in items:
            y = load_curve(run_dir, fname)[idx]
            color = cmap(0.45 + 0.55 * i / 4.0)
            ax.plot(t, y, color=color, lw=1.1, label=short, alpha=0.95)
            i += 1
    # stage boundaries (B/M/E)
    for b in (42, 85):
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_xticks([])
    m = mse_per_sample[idx]
    ax.set_title(f"idx={idx} gt={labels[idx].tolist()} mse(c28)={m:.1f}", fontsize=8)


fig, axes = plt.subplots(4, 2, figsize=(14, 14))
for k, idx in enumerate(rescued_pick):
    plot_case(axes[k, 0], int(idx))
axes[0, 0].set_title(f"RESCUED: pure wrong -> c28 correct (idx={rescued_pick[0]})", fontsize=10)
for k, idx in enumerate(broken_pick):
    plot_case(axes[k, 1], int(idx))
axes[0, 1].set_title(f"BROKEN: pure correct -> c28 wrong (idx={broken_pick[0]})", fontsize=10)
handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=8, fontsize=8)
fig.suptitle("verbalts_v2 seed1: GT vs pure vs applied steering (c2..c28)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "rescued_vs_broken.png", dpi=130)
plt.close(fig)

# MSE blowup cases
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, idx in zip(axes.ravel(), blowup_pick):
    plot_case(ax, int(idx))
handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=8, fontsize=8)
fig.suptitle("MSE blow-up cases under c28 (largest per-sample MSE)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "mse_blowup.png", dpi=130)
plt.close(fig)
print("saved ->", OUT)
