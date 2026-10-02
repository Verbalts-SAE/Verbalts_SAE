"""Plot middle single-peak failures with GT vs generated overlay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
STEER = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
OUT = ROOT / "results/electricity_v3/visualizations/verbalts_failure_diagnosis.png"

summary = json.loads((STEER / "summary.json").read_text())
indices = np.asarray(summary["indices"])
attrs = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_attrs_idx.npy")[indices]
gt = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_ts.npy")[indices, :, 0]
pure = np.load(STEER / "pure.npy")
caps = np.load(
    ROOT / "datasets/electricity_15min_semisynth_morph/valid_text_caps.npy", allow_pickle=True
).reshape(-1)[indices]

preds = json.loads((STEER / "pure_predictions.json").read_text())
sp = [[p["shape"] for p in c] for c in preds]

msp = [i for i in range(len(gt)) if attrs[i, 1] == 1]
msp_wrong = [i for i in msp if sp[i][1] != "single peak"]

# also show one "nothing over-oscillation" case: a nothing segment that got a false peak
false_peak = [i for i in range(len(gt)) if attrs[i, 1] == 0 and sp[i][1] in ("single peak", "double peaks", "sag")]

cases = msp_wrong[:3] + false_peak[:2]
fig, axes = plt.subplots(len(cases), 1, figsize=(11, 2.6 * len(cases)))
if len(cases) == 1:
    axes = [axes]
for row, i in enumerate(cases):
    ax = axes[row]
    ax.plot(gt[i], color="#1e7d32", lw=1.6, label="ground truth")
    ax.plot(pure[i], color="#c62828", lw=1.6, label="generated (pure)")
    for edge in (43, 86):
        ax.axvline(edge, color="gray", ls="--", lw=0.8)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks([0, 43, 86, 128])
    tag = "middle single peak missing" if i in msp_wrong else "false peak on nothing segment"
    ax.set_title(f"case {i} (valid {indices[i]}) — {tag}", fontsize=10)
    ax.set_xlabel(str(caps[i])[:110], fontsize=7.5, color="#333333")
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("VerbalTS failure diagnosis: narrow single peaks vanish, background over-oscillates",
             fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)

# numeric: generated peak location shift for wrong middle single peak
print("=== generated peak position in middle segment (wrong cases) ===")
for i in msp_wrong[:5]:
    seg = pure[i, 43:86]
    print(f"  case {i}: gen peak pos={np.argmax(seg)} (GT pos={np.argmax(gt[i, 43:86])})")

# numeric: how many 'nothing' middle segments get false shapes
fp = [i for i in range(len(gt)) if attrs[i, 1] == 0 and sp[i][1] != "nothing"]
print(f"\nnothing-middle misclassified as non-nothing: {len(fp)} / {sum(1 for i in range(len(gt)) if attrs[i,1]==0)}")
