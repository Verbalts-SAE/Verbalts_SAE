"""Is the whole-curve visual mismatch inherent to the task (uncontrolled background)
or a model deficiency?

Contrast experiment:
  - MSE(GT_i, GEN_i): generated curve vs its own GT
  - MSE(GT_i, GT_j): two DIFFERENT GT curves sharing the same caption (same attrs tuple)
If the two are comparable, the point-level mismatch is inherent: the caption does not
specify the background, so even the best model must 'invent' one. If GEN is much worse,
the model additionally degrades the distribution.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
STEER = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
D = ROOT / "datasets/electricity_15min_semisynth_morph"

summary = json.loads((STEER / "summary.json").read_text())
indices = np.asarray(summary["indices"])
gt = np.load(D / "valid_ts.npy")[indices, :, 0]
pure = np.load(STEER / "pure.npy")
attrs = np.load(D / "valid_attrs_idx.npy")

# full valid set for same-caption pairing
valid_ts = np.load(D / "valid_ts.npy")[:, :, 0]
valid_attrs = np.load(D / "valid_attrs_idx.npy")

# group valid samples by attrs tuple
from collections import defaultdict
groups = defaultdict(list)
for j in range(len(valid_ts)):
    groups[tuple(valid_attrs[j])].append(j)

mse_gt_gen, mse_gt_gt = [], []
for i in range(len(gt)):
    g = gt[i]
    mse_gt_gen.append(float(np.mean((g - pure[i]) ** 2)))
    # pick the same-caption valid sample closest to index (excluding itself)
    key = tuple(attrs[i])
    cands = groups[key]
    if len(cands) < 2:
        continue
    j = cands[0] if cands[0] != indices[i] else cands[1]
    mse_gt_gt.append(float(np.mean((g - valid_ts[j]) ** 2)))

mse_gt_gen = np.asarray(mse_gt_gen)
mse_gt_gt = np.asarray(mse_gt_gt)
print(f"n = {len(mse_gt_gt)} paired samples")
print(f"MSE(GT, GEN)  mean {mse_gt_gen.mean():.3f}  median {np.median(mse_gt_gen):.3f}")
print(f"MSE(GT, GT')  mean {mse_gt_gt.mean():.3f}  median {np.median(mse_gt_gt):.3f}   (same caption, different background)")
print(f"ratio GEN/GT'  mean {mse_gt_gen.mean()/mse_gt_gt.mean():.2f}")

# correlation comparison
corr_gt_gen, corr_gt_gt = [], []
for i in range(len(gt)):
    g = gt[i]
    corr_gt_gen.append(float(np.corrcoef(g, pure[i])[0, 1]))
    key = tuple(attrs[i])
    cands = groups[key]
    if len(cands) < 2:
        continue
    j = cands[0] if cands[0] != indices[i] else cands[1]
    corr_gt_gt.append(float(np.corrcoef(g, valid_ts[j])[0, 1]))
corr_gt_gen = np.asarray(corr_gt_gen)
corr_gt_gt = np.asarray(corr_gt_gt)
print(f"corr(GT, GEN) mean {corr_gt_gen.mean():.3f}")
print(f"corr(GT, GT') mean {corr_gt_gt.mean():.3f}")

# per-shape-combination: does GEN match the *average* GT better than individual GT?
print("\n=== whole-curve distance to same-caption GT CLUSTER CENTROID ===")
d_gt_centroid, d_gen_centroid = [], []
for i in range(len(gt)):
    key = tuple(attrs[i])
    cands = groups[key]
    centroid = valid_ts[cands].mean(axis=0)
    d_gt_centroid.append(float(np.mean((gt[i] - centroid) ** 2)))
    d_gen_centroid.append(float(np.mean((pure[i] - centroid) ** 2)))
d_gt_centroid = np.asarray(d_gt_centroid)
d_gen_centroid = np.asarray(d_gen_centroid)
print(f"MSE(GT_i, centroid)   mean {d_gt_centroid.mean():.3f}")
print(f"MSE(GEN_i, centroid)  mean {d_gen_centroid.mean():.3f}")
print(f"=> generated curves are {d_gen_centroid.mean()/d_gt_centroid.mean():.2f}x farther from their caption-cluster centroid than real GT curves")

# spectrum / roughness comparison
print("\n=== roughness (mean abs first-difference) ===")
gt_rough = np.mean(np.abs(np.diff(gt, axis=1)))
gen_rough = np.mean(np.abs(np.diff(pure, axis=1)))
tr_rough = np.mean(np.abs(np.diff(np.load(D / "train_ts.npy")[:, :, 0], axis=1)))
va_rough = np.mean(np.abs(np.diff(valid_ts, axis=1)))
print(f"train {tr_rough:.3f}  valid {va_rough:.3f}  GEN {gen_rough:.3f}  (GEN/train = {gen_rough/tr_rough:.2f})")
