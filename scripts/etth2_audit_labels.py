"""Audit etth2_2class label quality with the per-seed CNNs.

Questions for the dataset-purification decision (t3):
1. CNN-vs-label agreement per split per class (noise ceiling).
2. Per-seed CNN double-peaks recall/precision on the test set.
3. Morphological quality of double-peaks segments: peak-height ratio of the
   two dominant peaks; flag "weak double peak" segments whose smaller peak is
   < 0.3x the larger one (likely ambiguous for both CNN and generators).

Runs on CPU. Writes JSON report to results/etth2_2class/diag/label_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.shapes import classify_curves

DATA = ROOT / "datasets/etth2_2class/etth2_2class_dataset"
RES = ROOT / "results/etth2_2class"
SEGMENT_LEN = 43
SEGMENTS = ((0, 43), (43, 86), (85, 128))  # beginning/middle/end (CNN window 43)


def two_peak_ratio(seg: np.ndarray) -> float:
    """Ratio of the two dominant local maxima of a segment (1.0 if equal)."""
    x = np.asarray(seg, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return 1.0
    x = x - x.min()
    if x.max() <= 0:
        return 1.0
    x = x / x.max()
    # Local maxima with prominence >= 0.15 of range.
    peaks = []
    for i in range(1, x.size - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append((float(x[i]), i))
    peaks = [p for p in peaks if p[0] >= 0.15]
    if len(peaks) < 2:
        return 1.0
    peaks.sort(key=lambda p: -p[0])
    hi, lo = peaks[0][0], peaks[1][0]
    return hi / max(lo, 1e-6)


def main() -> None:
    report: dict = {"seeds": {}}
    out_dir = RES / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        curves = np.load(DATA / f"{split}_ts.npy")[:, :, 0]
        labels = np.load(DATA / f"{split}_labels.npy")  # (N, 3) int64
        report[f"{split}_n"] = int(len(curves))
        report[f"{split}_label_share_dp"] = float((labels == 1).mean())
        # Morphological weak-double-peak census (label-independent, all segments).
        weak = 0
        dp_seg_ratios = []
        sp_seg_ratios = []
        for i in range(len(curves)):
            for s, (a, b) in enumerate(SEGMENTS):
                ratio = two_peak_ratio(curves[i, a:b])
                if labels[i, s] == 1:
                    dp_seg_ratios.append(ratio)
                    if ratio > 3.0:
                        weak += 1
                else:
                    sp_seg_ratios.append(ratio)
        report[f"{split}_weak_dp_segments_ratio_gt3"] = (
            float(weak / max(len(dp_seg_ratios), 1))
        )
        report[f"{split}_dp_ratio_quantiles"] = [
            float(np.quantile(dp_seg_ratios, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        ]
        report[f"{split}_sp_ratio_quantiles"] = [
            float(np.quantile(sp_seg_ratios, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        ]
    for seed in (1, 7, 42):
        cnn = PeakValleyClassifier1D(segment_len=SEGMENT_LEN)
        cnn.load_state_dict(
            torch.load(RES / f"seed{seed}/cnn/segment_cnn.pth", map_location="cpu", weights_only=True)
        )
        cnn.eval()
        entry: dict = {}
        for split in ("train", "valid", "test"):
            curves = np.load(DATA / f"{split}_ts.npy")[:, :, 0]
            labels = np.load(DATA / f"{split}_labels.npy")
            preds = classify_curves(cnn, curves, torch.device("cpu"))
            # preds: list[N][3] of dicts with "shape" key from the 4-class vocab.
            seg_shapes = np.asarray([[d["shape"] for d in row] for row in preds])
            mapped = np.where(seg_shapes == "single peak", 0,
                              np.where(seg_shapes == "double peaks", 1, -1))
            agree = (mapped == labels).all(axis=1)
            dp_mask = labels[:, 0] == 1  # per-sample: beginning segment is double peaks
            sp_mask = labels[:, 0] == 0
            per_seg_agree = (mapped == labels).mean(0)
            dp_recall_all = (mapped[labels == 1] == 1).mean()
            dp_precision_all = (labels[mapped == 1] == 1).mean() if (mapped == 1).any() else 0.0
            entry[f"{split}_sample_agree"] = float(agree.mean())
            entry[f"{split}_per_segment_agree"] = per_seg_agree.tolist()
            entry[f"{split}_dp_recall"] = float(dp_recall_all)
            entry[f"{split}_dp_precision"] = float(dp_precision_all)
            entry[f"{split}_dp_sample_agree"] = float(agree[dp_mask].mean())
            entry[f"{split}_sp_sample_agree"] = float(agree[sp_mask].mean())
        report["seeds"][str(seed)] = entry
        print(f"seed{seed} test sample agree={entry['test_sample_agree']:.4f} "
              f"dp_recall={entry['test_dp_recall']:.4f} "
              f"dp_precision={entry['test_dp_precision']:.4f}")
    (out_dir / "label_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
