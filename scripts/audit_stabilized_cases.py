"""Recompute CNN labels and plot deterministic, explicitly selected audit cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.shapes import SHAPE_NAMES, classify_curves, parse_segment_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run, data = args.run_dir, args.data_root
    out = run / "visual_audit"
    out.mkdir(exist_ok=True)
    steer = run / "steering"
    summary = json.loads((steer / "summary.json").read_text())
    indices = np.asarray(summary["indices"])
    targets = np.asarray(SHAPE_NAMES)[np.load(data / "valid_attrs_idx.npy")[indices]]
    captions = np.load(data / "valid_text_caps.npy", allow_pickle=True).reshape(-1)[indices]
    assert np.array_equal(targets, np.asarray([parse_segment_shapes(c) for c in captions]))
    keys = ["pure", "sae", "dynamic_k32_e5120_g6", "dense_k0_e5120_g6"]
    titles = ["GT", "Pure", "SAE only", "Dynamic k32", "Dense"]
    curves = {"GT": np.load(data / "valid_ts.npy")[indices, :, 0]}
    curves.update({k: np.load(steer / f"{k}.npy") for k in keys})
    cnn = PeakValleyClassifier1D(segment_len=43)
    cnn.load_state_dict(torch.load(run / "cnn/segment_cnn.pth", map_location="cpu", weights_only=True))
    torch.set_num_threads(4)
    predictions, labels, scores = {}, {}, {}
    for key, values in curves.items():
        assert values.shape == (len(indices), 128) and np.isfinite(values).all()
        predictions[key] = classify_curves(cnn, values, torch.device("cpu"))
        labels[key] = np.asarray([[p["shape"] for p in row] for row in predictions[key]])
        scores[key] = (labels[key] == targets).sum(1)
        if key != "GT":
            saved = json.loads((steer / f"{key}_predictions.json").read_text())
            assert labels[key].tolist() == [[p["shape"] for p in row] for row in saved]
            assert np.isclose((scores[key] / 3).mean(), summary["variants"][key]["cnn"]["segment_accuracy"])
    delta = np.abs(curves[keys[2]] - curves["sae"])
    mae = delta.mean(1)
    correct = np.flatnonzero(scores["pure"] == 3)
    wrong = np.flatnonzero(scores["pure"] < 3)
    groups = {
        "cnn_correct_first6": correct[:6],
        "cnn_wrong_first6": wrong[:6],
        "steering_largest_change": np.argsort(-mae, kind="stable")[:6],
    }
    audit = {
        "selection": "First six in saved evaluation order; steering cases ranked by dynamic-vs-SAE MAE, descending.",
        "cnn_recomputed_on_cpu_matches_saved_labels": True,
        "caption_matches_attrs": True,
        "counts": {k: {"correct_segments": int(scores[k].sum()), "exact_curves": int((scores[k] == 3).sum())} for k in curves},
        "dynamic_vs_sae": {
            "mean_abs_delta": float(delta.mean()), "max_abs_delta": float(delta.max()),
            "per_curve_mae_quantiles": np.quantile(mae, [0, .5, .9, 1]).tolist(),
            "changed_segment_labels": int((labels[keys[2]] != labels["sae"]).sum()),
        },
        "cases": {},
    }
    short = {"nothing": "N", "single peak": "P", "double peaks": "D", "sag": "S"}
    def label_text(row):
        return " / ".join(short.get(x, x) for x in row)

    for group, positions in groups.items():
        audit["cases"][group] = []
        fig, axes = plt.subplots(len(positions), 5, figsize=(20, 3 * len(positions)), squeeze=False)
        for row, pos in enumerate(positions):
            audit["cases"][group].append({
                "position": int(pos), "valid_index": int(indices[pos]),
                "target": targets[pos].tolist(), "caption": str(captions[pos]),
                "predictions": {k: predictions[k][pos] for k in curves},
                "dynamic_vs_sae_mae": float(mae[pos]),
            })
            values = np.concatenate([v[pos] for v in curves.values()])
            lo, hi = float(values.min()), float(values.max())
            pad = max((hi - lo) * .12, .1)
            for col, (key, title) in enumerate(zip(curves, titles)):
                ax = axes[row, col]
                ax.plot(curves[key][pos], color="#2166ac", lw=1.4)
                if key == keys[2]:
                    ax.plot(curves["sae"][pos], color="#e66101", lw=1, ls="--", label="SAE overlay")
                    ax.legend(fontsize=7, loc="upper right")
                for edge in (42.5, 85.5):
                    ax.axvline(edge, color="gray", ls=":", lw=.8)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_xticks([0, 43, 85, 127])
                ax.grid(alpha=.15)
                ax.set_title(f"{title}: CNN {scores[key][pos]}/3", fontsize=10)
                ax.set_xlabel(f"CNN: {label_text(labels[key][pos])}", fontsize=9)
            axes[row, 0].set_ylabel(f"row {pos}, valid {indices[pos]}\nTarget: {label_text(targets[pos])}", fontsize=9)
        fig.suptitle(f"{group} | N=nothing, P=single peak, D=double peaks, S=sag\n"
                     "CNN agreement is NOT human verification. Shared y-axis within each row; no clipping.\n"
                     "CNN windows: [0:43], [43:86], [85:128] (last two overlap at 85).", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(out / f"{group}.png", dpi=130)
        plt.close(fig)

    # Dedicated magnified difference plot: do not mistake SAE insertion for steering.
    positions = groups["steering_largest_change"][:3]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    for row, pos in enumerate(positions):
        axes[row, 0].plot(curves["sae"][pos], label="SAE", color="#e66101")
        axes[row, 0].plot(curves[keys[2]][pos], label="Dynamic", ls="--", color="#2166ac")
        axes[row, 0].legend()
        axes[row, 0].set_title(f"valid {indices[pos]}: MAE={mae[pos]:.6f}")
        for key in (keys[2], keys[3]):
            axes[row, 1].plot(curves[key][pos] - curves["sae"][pos], label=f"{key.split('_')[0]} - SAE")
        axes[row, 1].legend()
        axes[row, 1].axhline(0, color="gray", lw=.7)
        axes[row, 1].set_title("Difference (separate vertical scale)")
    fig.tight_layout()
    fig.savefig(out / "steering_delta.png", dpi=140)
    plt.close(fig)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "cases"}, indent=2))
    print(out)


if __name__ == "__main__":
    main()