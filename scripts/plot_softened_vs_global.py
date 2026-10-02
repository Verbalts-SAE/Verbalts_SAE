"""Compare V2, V3, and the stabilized V3 generation when available."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
REF = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
V2 = ROOT / "results/electricity_v3/quality/valid128_global_captions"
V3 = ROOT / "results/electricity_v3/quality/valid128_softened_captions"
V3_STABLE = ROOT / "results/electricity_v3/quality/valid128_softened_captions_stabilized"
OUT = ROOT / "results/electricity_v3/visualizations/softened_vs_global_comparison.png"

summary = json.loads((REF / "summary.json").read_text())
indices = np.asarray(summary["indices"])
attrs = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_attrs_idx.npy")[indices]
truth = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_ts.npy")[indices, :, 0]
old_c = np.load(REF / "pure.npy")
v2_c = np.load(V2 / "pure.npy")
v3_c = np.load(V3 / "pure.npy")
v3_stable_path = V3_STABLE / "pure.npy"
v3_stable_c = np.load(v3_stable_path) if v3_stable_path.is_file() else None
caps = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_text_caps.npy",
               allow_pickle=True).reshape(-1)[indices]
M = ["nothing", "single peak", "double peaks", "sag"]


def rough(x):
    return float(np.mean(np.abs(np.diff(x))))


def soft_count(c):
    p = str(c).split("global pattern has ")[-1]
    return sum(1 for w in ["fairly high", "fairly brisk", "fairly pronounced"] if w in p)


sc = np.array([soft_count(c) for c in caps])
v3_r = np.array([rough(x) for x in v3_c])
v2_r = np.array([rough(x) for x in v2_c])

# pick cases
# 1. triple-fairly sample where V3 exploded (biggest V3 - V2 roughness gap)
sel3 = np.where(sc == 3)[0]
case_explode = int(sel3[np.argmax(v3_r[sel3] - v2_r[sel3])])

# 2. case 7 (user's earlier concern) under V3
case_7 = 7

# 3. a count=0 sample where V3 stays healthy
sel0 = np.where(sc == 0)[0]
case_ok = int(sel0[np.argmin(np.abs(v3_r[sel0] - np.median(v3_r[sel0])))])

# 4. double peaks sample to inspect morphology
case_dp = None
for i in range(len(truth)):
    for st in range(3):
        if attrs[i, st] == 2:
            case_dp = i
            break
    if case_dp is not None:
        break

cases = [
    (case_explode, f"case {case_explode}: triple 'fairly' wording (V3 exploded rough {v3_r[case_explode]:.2f} vs V2 {v2_r[case_explode]:.2f})"),
    (case_7, f"case 7: user-flagged roughness (V3 rough {v3_r[case_7]:.2f} vs V2 {v2_r[case_7]:.2f})"),
    (case_ok, f"case {case_ok}: no softened wording, V3 stays healthy"),
    (case_dp, f"case {case_dp}: double peaks morphology check"),
]

columns = [
    (truth, "#1e7d32", "ground truth"),
    (old_c, "#c62828", "OLD"),
    (v2_c, "#1565c0", "V2 global"),
    (v3_c, "#6a1b9a", "V3 softened"),
]
if v3_stable_c is not None:
    columns.append((v3_stable_c, "#ef6c00", "V3 stabilized"))

fig, axes = plt.subplots(4, len(columns), figsize=(4.5 * len(columns), 11), squeeze=False)
for row, (ci, title) in enumerate(cases):
    for col, (curves, color, label) in enumerate(columns):
        curve = curves[ci]
        ax = axes[row, col]
        ax.plot(curve, color=color, lw=1.5, label=label)
        for edge in (43, 86):
            ax.axvline(edge, color="gray", ls="--", lw=0.7)
        ax.set_ylim(-6.0, 6.0)
        ax.set_xticks([0, 43, 86, 128])
        if col == 0:
            ax.set_ylabel(f"GT: {'/'.join(M[v] for v in attrs[ci])}", fontsize=8)
        if row == 0:
            ax.set_title(label, fontsize=10, color=color)
        if row == len(cases) - 1:
            ax.set_xlabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=6)
    axes[row, 0].set_ylabel(f"{title}\nGT: {'/'.join(M[v] for v in attrs[ci])}", fontsize=7)
    axes[row, 0].yaxis.set_label_coords(-0.13, 0.5)

fig.suptitle("GT vs OLD vs V2 global vs V3 softened vs stabilized DDIM (when available)",
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
print()
print("captions of the shown cases (V3 wording):")
for ci, _ in cases:
    print(f"  case {ci}: {str(caps[ci])[:150]}")
print()
print("summary stats:")
print(f"  triple-fairly (n={int((sc==3).sum())}): V2 rough {v2_r[sc==3].mean():.3f} -> V3 {v3_r[sc==3].mean():.3f}")
print(f"  count=0 (n={int((sc==0).sum())}):      V2 rough {v2_r[sc==0].mean():.3f} -> V3 {v3_r[sc==0].mean():.3f}")
if v3_stable_c is not None:
    stable_r = np.array([rough(x) for x in v3_stable_c])
    ci = case_explode
    print(
        f"  case {ci} stabilization: rough {v3_r[ci]:.3f} -> {stable_r[ci]:.3f}; "
        f"range [{v3_c[ci].min():.3f}, {v3_c[ci].max():.3f}] -> "
        f"[{v3_stable_c[ci].min():.3f}, {v3_stable_c[ci].max():.3f}]"
    )
