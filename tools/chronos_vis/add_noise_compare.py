#!/usr/bin/env python
"""Add small Gaussian noise to stratified windows and compare visually.

Noise levels relative to each window's std: 0 (original), 0.05, 0.10.
One 9x11 sample grid per level (same windows, same layout).
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/public/home/liym2024/Verbalts_SAE/results/trend_windows_ma65_screened_stratified"
DIR = os.path.join(BASE_DIR, "electricity_15min_ma65_w128_stratified")
OUT = os.path.join(DIR, "noise_compare")
os.makedirs(OUT, exist_ok=True)

NOISE_LEVELS = [0.0, 0.05, 0.10]
SEED = 42


def main():
    W = np.load(os.path.join(DIR, "windows.npy"))
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal(W.shape)  # shared base noise, scaled per level

    fig, axes = plt.subplots(len(NOISE_LEVELS), 11,
                             figsize=(24, 3.1 * len(NOISE_LEVELS)),
                             constrained_layout=True)
    pick = np.linspace(0, len(W) - 1, 11).astype(int)
    for r, level in enumerate(NOISE_LEVELS):
        stds = W.std(axis=1, keepdims=True)
        Wn = W + level * stds * noise
        for c, i in enumerate(pick):
            ax = axes[r, c]
            ax.plot(Wn[i], lw=0.8)
            ax.set_title(f"#{i}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xticks([])
        axes[r, 0].set_ylabel(f"noise={level:.2f}*std", fontsize=10)
    fig.suptitle("same 11 windows under different noise levels", fontsize=13)
    fig.savefig(os.path.join(OUT, "noise_level_compare.png"), dpi=120)
    plt.close(fig)

    # full stratified grid for the two noisy levels
    for level in [0.05, 0.10]:
        stds = W.std(axis=1, keepdims=True)
        Wn = W + level * stds * noise
        fig, axes = plt.subplots(9, 11, figsize=(22, 16),
                                 constrained_layout=True)
        for b in range(9):
            for c in range(11):
                i = b * 11 + c
                axes[b, c].plot(Wn[i], lw=0.7)
                axes[b, c].tick_params(labelsize=5)
                axes[b, c].set_xticks([])
        fig.suptitle(f"stratified windows + noise {level:.2f}*std "
                     f"(first 99 of {len(W):,})", fontsize=13)
        fig.savefig(os.path.join(OUT, f"grid_noise_{level:.2f}.png"), dpi=120)
        plt.close(fig)

    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
