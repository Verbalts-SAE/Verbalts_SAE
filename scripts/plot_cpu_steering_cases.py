"""Plot steering case comparisons for the three cpu_shape_v1 models (test split).

Bridge/diffusets: recompute per-segment CNN predictions on saved curves.
Verbalts: reuse the saved per-sample predictions JSONs.
Output: results/cpu_shape_v1/steering/case_figs/{bridge,diffusets,verbalts}_cases.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
BENCH = Path("/public/home/liym2024/ConTSG-Bench")
STEER = ROOT / "results/cpu_shape_v1/steering"
DATA = ROOT / "datasets/cpu_shape_v1"
CNN_PT = ROOT / "results/cpu_shape_v1/pipeline/cnn/best.pt"
OUT = STEER / "case_figs"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ("nothing", "single peak", "double peaks", "sag")


def load_cnn():
    sys_path = str(BENCH)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from contsg.eval.metrics.segment import PeakValleyClassifier1D
    model = PeakValleyClassifier1D(segment_len=43)
    model.load_state_dict(torch.load(CNN_PT, map_location="cpu", weights_only=True))
    return model.eval()


def classify(curves: np.ndarray) -> list[list[str]]:
    import sys
    sys_path = str(BENCH)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from sae.visualize_variants import classify_curves
    cnn = load_cnn()
    preds = classify_curves(cnn, curves, torch.device("cpu"))
    return [[p["shape"] for p in c] for c in preds]


def truth_labels() -> list[list[str]]:
    attrs = np.load(DATA / "test_attrs_idx.npy")
    return [[NAMES[a] for a in row] for row in attrs]


def truth_curves() -> np.ndarray:
    return np.load(DATA / "test_ts.npy")[:, :, 0]


def captions() -> np.ndarray:
    return np.load(DATA / "test_text_caps.npy", allow_pickle=True).reshape(-1)


def select_cases(base_preds, steer_preds, truth, kind, n, offset=0):
    """kind: 'fixed' (baseline wrong -> steer right) or 'degraded'."""
    picks = []
    for i in range(len(truth)):
        base_ok = all(p == t for p, t in zip(base_preds[i], truth[i]))
        steer_ok = all(p == t for p, t in zip(steer_preds[i], truth[i]))
        if kind == "fixed" and (not base_ok) and steer_ok:
            picks.append(i)
        elif kind == "degraded" and base_ok and (not steer_ok):
            picks.append(i)
        if len(picks) >= offset + n:
            break
    return picks[offset:offset + n]


def draw_panel(ax, curve, truth_curve, preds, truth, title, ylim):
    ax.plot(truth_curve, color="#9e9e9e", lw=1.2, ls="--", alpha=0.8, label="truth")
    ax.plot(curve, color="#1f4e79", lw=1.5, label="generated")
    for edge in (43, 86):
        ax.axvline(edge, color="#bdbdbd", ls=":", lw=0.8)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 43, 86, 128])
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, pad=3)
    labels = []
    for pred, true in zip(preds, truth):
        if pred == true:
            labels.append(pred)
        else:
            labels.append(f"{pred} (T:{true})")
    ok = sum(p == t for p, t in zip(preds, truth))
    ax.set_xlabel(" | ".join(labels), fontsize=7.5,
                  color="#1e7d32" if ok == 3 else "#c62828")


def plot_model(name, rows, out_name):
    """rows: list of (caption, idx, [(subtitle, curves, preds), ...])"""
    nrows = len(rows)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    gt = truth_curves()
    truth = truth_labels()
    for r, (cap, idx, panels) in enumerate(rows):
        ylim = (min(np.min(p[1][idx]) for p in panels) - 0.5,
                max(np.max(p[1][idx]) for p in panels) + 0.5)
        for c, (subtitle, curves, preds) in enumerate(panels):
            ax = axes[r][c]
            draw_panel(ax, curves[idx], gt[idx], preds[idx], truth[idx],
                       f"{subtitle}  (idx {idx})", ylim)
            if c == 0:
                ax.set_ylabel(cap[:44] + ("..." if len(cap) > 44 else ""),
                              fontsize=7.5, rotation=0, ha="right", va="center")
    fig.suptitle(f"cpu_shape_v1 test steering cases — {name}", fontsize=12)
    fig.tight_layout(rect=[0.12, 0, 1, 0.96])
    out = OUT / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


def main():
    truth = truth_labels()
    caps = captions()

    # ---- bridge: baseline vs relcap8_ptn (the +15.8pp winner) ----
    bdir = STEER / "bridge_test_1020364" / "seed1"
    base_b = np.load(bdir / "baseline/curves.npy")
    steer_b = np.load(bdir / "relcap8_ptn/curves.npy")
    pb = classify(base_b)
    ps = classify(steer_b)
    fixed = select_cases(pb, ps, truth, "fixed", 4)
    degr = select_cases(pb, ps, truth, "degraded", 1)
    print("bridge fixed idx:", fixed, "degraded idx:", degr)
    rows = [(caps[i], i, [("baseline (wrong)", base_b, pb),
                          ("relcap8_ptn (fixed)", steer_b, ps)]) for i in fixed]
    rows.append((caps[degr[0]], degr[0],
                 [("baseline (correct)", base_b, pb),
                  ("relcap8_ptn (broke it)", steer_b, ps)]))
    plot_model("Bridge", rows, "bridge_cases.png")

    # ---- diffusets: baseline vs champion t8_i2 ----
    ddir = STEER / "diffusets_test_1020478" / "seed1"
    base_d = np.load(ddir / "baseline/curves.npy")
    steer_d = np.load(ddir / "champion/curves.npy")
    pd_ = classify(base_d)
    pc = classify(steer_d)
    fixed_d = select_cases(pd_, pc, truth, "fixed", 2)
    same_d = [i for i in range(2000)
              if all(p == t for p, t in zip(pd_[i], truth[i]))
              and all(p == t for p, t in zip(pc[i], truth[i]))][:1]
    print("diffusets fixed idx:", fixed_d, "both-ok idx:", same_d)
    rows = [(caps[i], i, [("baseline (wrong)", base_d, pd_),
                          ("champion (fixed)", steer_d, pc)]) for i in fixed_d]
    rows.append((caps[same_d[0]], same_d[0],
                 [("baseline (correct)", base_d, pd_),
                  ("champion (unchanged)", steer_d, pc)]))
    plot_model("DiffuSETS", rows, "diffusets_cases.png")

    # ---- verbalts: pure vs dynamic_k8 (no steering gain) ----
    vdir = STEER / "verbalts_test_1020471" / "seed1"
    pure = np.load(vdir / "pure.npy")
    dyn8 = np.load(vdir / "dynamic_k8_e5120_g6.npy")
    pp = [[p["shape"] for p in c]
          for c in json.loads((vdir / "pure_predictions.json").read_text())]
    pk = [[p["shape"] for p in c]
          for c in json.loads((vdir / "dynamic_k8_e5120_g6_predictions.json").read_text())]
    ok_idx = [i for i in range(2000)
              if all(p == t for p, t in zip(pp[i], truth[i]))][:2]
    print("verbalts both-ok idx:", ok_idx)
    rows = [(caps[i], i, [("pure (correct)", pure, pp),
                          ("dynamic_k8 (no change)", dyn8, pk)]) for i in ok_idx]
    plot_model("VerbalTS", rows, "verbalts_cases.png")


if __name__ == "__main__":
    main()
