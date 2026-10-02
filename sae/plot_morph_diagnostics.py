"""Plot paired validation cases and timestep guidance diagnostics (no test data)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sae.shapes import SHAPE_NAMES
from sae.provenance import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    summary = json.loads((run / "summary.json").read_text())
    indices = np.asarray(summary["indices"])
    targets = np.asarray(SHAPE_NAMES)[np.load(args.data_root / "valid_attrs_idx.npy")[indices]]
    truth = np.load(args.data_root / "valid_ts.npy")[indices, :, 0]
    captions = np.load(args.data_root / "valid_text_caps.npy", allow_pickle=True).reshape(-1)[indices]
    variants = list(summary["variants"])
    curves = {v: np.load(run / f"{v}.npy") for v in variants}
    pred = {v: np.asarray([[s["shape"] for s in row] for row in
                          json.loads((run / f"{v}_predictions.json").read_text())]) for v in variants}
    correct = {v: (pred[v] == targets) for v in variants}
    candidates = [v for v in variants if v not in ("pure", "sae")]
    best = max(candidates, key=lambda v: correct[v].mean())
    out = run / "figures"
    out.mkdir(exist_ok=True)
    selected = []
    groups = {
        "improved_vs_sae": np.flatnonzero(correct[best].sum(1) > correct["sae"].sum(1)),
        "worsened_vs_sae": np.flatnonzero(correct[best].sum(1) < correct["sae"].sum(1)),
        "middle_failure": np.flatnonzero(~correct["reference"][:, 1]),
        "nothing_middle_failure": np.flatnonzero((targets[:, 1] == "nothing") & ~correct["reference"][:, 1]),
        "active_middle_fix": np.flatnonzero(correct["active_only"][:, 1] & ~correct["sae"][:, 1]),
        "active_middle_break": np.flatnonzero(~correct["active_only"][:, 1] & correct["sae"][:, 1]),
    }
    for reason, members in groups.items():
        for i in members[:2]:
            selected.append((reason, int(i)))
    shown = list(dict.fromkeys(["pure", "sae", "reference", "middle", "active_only", best]))
    records = []
    for case_number, (reason, i) in enumerate(selected):
        fig, axes = plt.subplots(len(shown), 1, figsize=(12, 2.25 * len(shown)), sharex=True, sharey=True)
        for ax, v in zip(axes, shown):
            ax.plot(truth[i], color="0.5", alpha=.7, label="Ground truth (not unique target)")
            ax.plot(curves[v][i], color="tab:blue", label=v)
            ax.axvspan(43, 85, color="orange", alpha=.07)
            for x in (42.5, 85.5):
                ax.axvline(x, color="0.6", linestyle=":")
            ax.set_title(f"{v} | CNN: {' / '.join(pred[v][i])}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=.2)
        fig.suptitle(f"valid index {indices[i]} | {reason}\nTarget: {' / '.join(targets[i])}")
        axes[-1].set_xlabel("Time index (original model scale; no per-curve rescaling)")
        fig.tight_layout(rect=(0, 0, 1, .96))
        filename = f"case_{case_number:02d}_{indices[i]}_{reason}.png"
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)
        records.append(dict(file=filename, valid_index=int(indices[i]), reason=reason,
                            caption=str(captions[i]), target=targets[i].tolist()))
    write_json(out / "cases.json", dict(best_by_validation_segment_accuracy=best, cases=records))
    diagnostics = {}
    for v in candidates:
        rows = json.loads((run / f"{v}_trace.json").read_text())
        ts = sorted({r["timestep"] for r in rows})
        keys = ["probability_before", "probability_after", "selected_cap_fraction",
                "selected_dead_fraction", "selected_relu_clip_fraction", "selected_zero_move_fraction"]
        means = {key: np.asarray([np.mean([r[key] for r in rows if r["timestep"] == t], axis=0)
                                  for t in ts]) for key in keys}
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        for h, stage in enumerate(("Beginning", "Middle", "End")):
            axes[0].plot(ts, means["probability_before"][:, h], label=f"{stage} before")
            axes[0].plot(ts, means["probability_after"][:, h], "--", label=f"{stage} after")
        for key in keys[2:]:
            axes[1].plot(ts, means[key], label=key.replace("selected_", ""))
        for ax in axes:
            ax.invert_xaxis()
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            ax.set_ylim(-.02, 1.02)
        axes[0].set_title(f"{v}: MLP target probability before/after one latent intervention")
        axes[1].set_xlabel("Diffusion timestep (generation direction: high to low)")
        fig.tight_layout()
        fig.savefig(out / f"trace_{v}.png", dpi=130)
        plt.close(fig)
        diagnostics[v] = {key: np.mean([r[key] for r in rows], axis=0).tolist() for key in keys}
        diagnostics[v]["mlp_accuracy_before"] = np.mean([
            np.asarray(r["prediction_before"]) == r["target"] for r in rows], axis=0).tolist()
        diagnostics[v]["middle_fixes_vs_sae"] = int((correct[v][:, 1] & ~correct["sae"][:, 1]).sum())
        diagnostics[v]["middle_breaks_vs_sae"] = int((~correct[v][:, 1] & correct["sae"][:, 1]).sum())
    write_json(out / "diagnostics.json", diagnostics)
    print(f"Saved {len(selected)} paired cases and {len(candidates)} trace plots to {out}")


if __name__ == "__main__":
    main()