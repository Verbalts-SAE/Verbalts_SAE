"""Quantitative diagnosis of VerbalTS generation failures (numpy only)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
STEER = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
NAMES = ("nothing", "single peak", "double peaks", "sag")
EDGES = [(0, 43), (43, 86), (85, 128)]

summary = json.loads((STEER / "summary.json").read_text())
indices = np.asarray(summary["indices"])
attrs = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_attrs_idx.npy")[indices]
gt = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_ts.npy")[indices, :, 0]
pure = np.load(STEER / "pure.npy")


def load_shapes(name):
    preds = json.loads((STEER / f"{name}_predictions.json").read_text())
    return [[p["shape"] for p in c] for c in preds]


sp = load_shapes("pure")


def seg_stats(curve):
    """Per-segment: peak height (max-median), valley depth (median-min), range."""
    out = []
    for a, b in EDGES:
        seg = curve[a:b]
        med = np.median(seg)
        out.append((float(np.max(seg) - med), float(med - np.min(seg)),
                    float(np.max(seg) - np.min(seg))))
    return out


print("=== per-shape segment amplitude: GT vs generated (pure) ===")
for shape_id, shape in enumerate(NAMES):
    g_peak, g_range, p_peak, p_range, n = [], [], [], [], 0
    for i in range(len(gt)):
        for stage in range(3):
            if attrs[i, stage] != shape_id:
                continue
            n += 1
            g = seg_stats(gt[i])[stage]
            p = seg_stats(pure[i])[stage]
            g_peak.append(g[0]); g_range.append(g[2])
            p_peak.append(p[0]); p_range.append(p[2])
    if n == 0:
        continue
    print(f"{shape:<13} n={n:4d} | GT peak {np.mean(g_peak):.3f} range {np.mean(g_range):.3f}"
          f" | GEN peak {np.mean(p_peak):.3f} range {np.mean(p_range):.3f}"
          f" | ratio peak {np.mean(p_peak)/max(np.mean(g_peak),1e-9):.2f}")

print()
print("=== middle single peak failures: what does the generated middle look like? ===")
msp = [i for i in range(len(gt)) if attrs[i, 1] == 1]  # middle single peak
msp_wrong = [i for i in msp if sp[i][1] != "single peak"]
print(f"middle single peak total: {len(msp)}, misclassified: {len(msp_wrong)}")
for i in msp_wrong[:6]:
    g = seg_stats(gt[i])[1]
    p = seg_stats(pure[i])[1]
    print(f"  case {i}: GT middle peak={g[0]:.3f} range={g[2]:.3f} "
          f"| GEN middle peak={p[0]:.3f} range={p[2]:.3f} "
          f"| predicted={sp[i][1]}")

print()
print("=== where is the middle single peak located in GT? (peak position inside segment) ===")
positions = []
for i in msp:
    seg = gt[i, 43:86]
    positions.append(int(np.argmax(seg)))
print("GT middle single peak positions (within 43-len segment): min %d, mean %.1f, max %d"
      % (min(positions), np.mean(positions), max(positions)))

# per-stage confusion: what does middle single peak get predicted as?
print()
print("=== middle single peak -> predicted distribution ===")
from collections import Counter
c = Counter(sp[i][1] for i in msp)
print(dict(c))

# also check beginning single peak and end single peak recall for comparison
print()
print("=== per-stage single peak recall ===")
for stage, name in enumerate(["Beginning", "Middle", "End"]):
    ids = [i for i in range(len(gt)) if attrs[i, stage] == 1]
    acc = sum(1 for i in ids if sp[i][stage] == "single peak") / len(ids)
    print(f"{name}: n={len(ids)}, recall={acc:.3f}")
