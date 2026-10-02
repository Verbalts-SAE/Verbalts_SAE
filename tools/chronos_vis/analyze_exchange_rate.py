#!/usr/bin/env python
"""Deep-dive analysis of the exchange_rate chronos arrow dataset.

Outputs:
  - printed stats table: per-channel time span, mean/std/min/max, skew,
    lag-1 autocorrelation, ADF tau (if statsmodels available)
  - figs/exchange_rate/stats_overview.png     : 8 full channels w/ real dates
  - figs/exchange_rate/corr_matrix.png        : Pearson corr (levels + log returns)
  - figs/exchange_rate/windows128.png         : 128-point windows per channel
  - figs/exchange_rate/returns.png            : log-return distributions
"""
import os

import numpy as np
import pyarrow as pa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "exchange_rate")
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(BASE, "exchange_rate", "train-00000-of-00001.arrow")

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_ADF = True
except Exception:
    HAS_ADF = False


def adf_tau(y):
    """Fallback ADF-like tau on y (no constant, 1 lag) via numpy lstsq."""
    dy = np.diff(y)
    if len(dy) < 3:
        return np.nan
    x = np.column_stack([y[:-1], dy[:-1]])
    try:
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(dy) - 1), x[1:]]),
                                   dy[1:], rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    resid = dy[1:] - np.column_stack([np.ones(len(dy) - 1), x[1:]]) @ beta
    dof = len(resid) - 3
    if dof <= 0:
        return np.nan
    sigma2 = resid @ resid / dof
    xx = np.column_stack([np.ones(len(dy) - 1), x[1:]])
    cov = sigma2 * np.linalg.inv(xx.T @ xx)
    return beta[1] / np.sqrt(cov[1, 1])


def main():
    table = pa.ipc.open_stream(PATH).read_all()
    ids = table.column("id").to_pylist()
    ts0 = table.column("timestamp")[0].as_py()
    print(f"rows={table.num_rows}, ids={ids}")
    print(f"first ts={ts0[0]}, last ts={ts0[-1]}, n={len(ts0)}")

    series = {}
    for i in range(table.num_rows):
        s = np.asarray(table.column("target")[i].as_py(), dtype=np.float64)
        series[ids[i]] = s[np.isfinite(s)]

    # ---- stats table ----
    print("\nchannel      len   mean     std      min      max      skew    lag1    adf_tau")
    for cid, s in series.items():
        skew = float(((s - s.mean()) ** 3).mean() / (s.std() ** 3 + 1e-12))
        lag1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        tau = float(adfuller(s, autolag="AIC")[0]) if HAS_ADF else float(adf_tau(s))
        print(f"{cid:11s} {len(s):6d}  {s.mean():8.4f} {s.std():7.4f} "
              f"{s.min():8.4f} {s.max():8.4f} {skew:7.3f} {lag1:6.3f} {tau:9.3f}")

    n = len(series)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (cid, s) in zip(axes.flat, series.items()):
        ax.plot(ts0, s, lw=0.7)
        ax.set_title(f"{cid}", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("exchange_rate — all 8 channels (daily)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "stats_overview.png"), dpi=110)
    plt.close(fig)

    # ---- correlation matrices ----
    mat = np.column_stack([series[c] for c in ids])
    lmat = np.column_stack([np.diff(np.log(series[c])) for c in ids])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, m, title in [(axes[0], np.corrcoef(mat.T), "levels (Pearson)"),
                         (axes[1], np.corrcoef(lmat.T), "log returns (Pearson)")]:
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n), ids, rotation=45, fontsize=8)
        ax.set_yticks(range(n), ids, fontsize=8)
        for r in range(n):
            for c in range(n):
                ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center",
                        fontsize=7)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("exchange_rate — channel correlation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "corr_matrix.png"), dpi=110)
    plt.close(fig)

    # ---- 128-point windows (align with synth-u length) ----
    W = 128
    n_win = 4
    fig, axes = plt.subplots(n, n_win, figsize=(n_win * 3.4, n * 2.4))
    for row, (cid, s) in enumerate(series.items()):
        starts = np.linspace(0, len(s) - W, n_win).astype(int)
        for col_i, st in enumerate(starts):
            ax = axes[row, col_i]
            ax.plot(s[st:st + W], lw=0.9)
            ax.set_title(f"{cid} [{st}:{st + W}]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_visible(False)
    fig.suptitle("exchange_rate — 128-point windows (synth-u aligned length)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "windows128.png"), dpi=110)
    plt.close(fig)

    # ---- log returns ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (cid, s) in zip(axes.flat, series.items()):
        r = np.diff(np.log(s))
        ax.hist(r, bins=120, color="tab:blue", alpha=0.85)
        ax.set_title(f"{cid} log-return (std={r.std():.4f})", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("exchange_rate — daily log-return distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "returns.png"), dpi=110)
    plt.close(fig)
    print("\nfigures saved to", OUT)


if __name__ == "__main__":
    main()
