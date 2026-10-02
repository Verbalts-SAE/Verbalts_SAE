"""Side-by-side: GT vs OLD (synth-u style) vs NEW (global caption) generations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
REF = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
NEW = ROOT / "results/electricity_v3/quality/valid128_global_captions"
OUT = ROOT / "results/electricity_v3/visualizations/global_caption_comparison.png"

summary = json.loads((REF / "summary.json").read_text())
indices = np.asarray(summary["indices"])
attrs = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_attrs_idx.npy")[indices]
truth = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_ts.npy")[indices, :, 0]
old_c = np.load(REF / "pure.npy")
new_c = np.load(NEW / "pure.npy")
caps = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_text_caps.npy",
               allow_pickle=True).reshape(-1)[indices]
M = ["nothing", "single peak", "double peaks", "sag"]

old_p = json.loads((REF / "pure_predictions.json").read_text())
new_p = json.loads((NEW / "pure_predictions.json").read_text())
old_s = [[p["shape"] for p in c] for c in old_p]
new_s = [[p["shape"] for p in c] for c in new_p]

def rough(x):
    return float(np.mean(np.abs(np.diff(x))))

def attrs_str(a):
    return "/".join(M[v] for v in a)

# --- pick representative cases ---
# 1. biggest roughness improvement (visual quality)
rough_delta = [rough(old_c[i]) - rough(new_c[i]) for i in range(len(truth))]
case_rough = int(np.argmax(rough_delta))

# 2. double peaks lost by NEW (OLD right, NEW wrong)
case_dp = None
for i in range(len(truth)):
    for st in range(3):
        if attrs[i, st] == 2 and old_s[i][st] == "double peaks" and new_s[i][st] != "double peaks":
            case_dp = i
            break
    if case_dp is not None:
        break

# 3. single peak gained by NEW (OLD wrong, NEW right)
case_sp = None
for i in range(len(truth)):
    for st in range(3):
        if attrs[i, st] == 1 and old_s[i][st] != "single peak" and new_s[i][st] == "single peak":
            case_sp = i
            break
    if case_sp is not None:
        break

# 4. sag lost by NEW
case_sag = None
for i in range(len(truth)):
    for st in range(3):
        if attrs[i, st] == 3 and old_s[i][st] == "sag" and new_s[i][st] != "sag":
            case_sag = i
            break
    if case_sag is not None:
        break

cases = [
    (case_rough, f"case {case_rough}: background roughness fixed (old rough {rough(old_c[case_rough]):.2f} -> new {rough(new_c[case_rough]):.2f})"),
    (case_dp, f"case {case_dp}: double peaks lost by global caption"),
    (case_sp, f"case {case_sp}: single peak gained by global caption"),
    (case_sag, f"case {case_sag}: sag lost by global caption"),
]

def short(s):
    return {"nothing": "n", "single peak": "sp", "double peaks": "dp", "sag": "sag"}[s]

fig, axes = plt.subplots(4, 3, figsize=(15, 11))
for row, (ci, title) in enumerate(cases):
    for col, (curve, color, label) in enumerate([
        (truth[ci], "#1e7d32", "ground truth"),
        (old_c[ci], "#c62828", "old caption"),
        (new_c[ci], "#1565c0", "new global caption"),
    ]):
        ax = axes[row, col]
        ax.plot(curve, color=color, lw=1.5, label=label)
        for edge in (43, 86):
            ax.axvline(edge, color="gray", ls="--", lw=0.7)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xticks([0, 43, 86, 128])
        if col == 0:
            ax.set_ylabel(f"GT: {attrs_str(attrs[ci])}", fontsize=8)
        if col == 1:
            ax.set_title(f"OLD pred: {'/'.join(short(s) for s in old_s[ci])}", fontsize=8, color="#c62828")
        if col == 2:
            ax.set_title(f"NEW pred: {'/'.join(short(s) for s in new_s[ci])}", fontsize=8, color="#1565c0")
        if row == 3:
            ax.set_xlabel(label, fontsize=9)
        if row == 0 and col == 0:
            ax.set_title("ground truth", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    axes[row, 0].set_ylabel(f"GT: {attrs_str(attrs[ci])}\n{title}", fontsize=8)
    axes[row, 0].yaxis.set_label_coords(-0.12, 0.5)

fig.suptitle("GT vs OLD (synth-u style caption) vs NEW (morphology + global background description)",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
print("captions:")
for ci, _ in cases:
    print(f"  case {ci}: {str(caps[ci])[:130]}")
