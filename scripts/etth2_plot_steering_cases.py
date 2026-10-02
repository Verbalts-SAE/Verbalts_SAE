"""Plot representative etth2_2class steering cases (no torch import).

Reads the on-policy steering outputs for a seed, compares pure vs steered
curves against the ground-truth curve, and saves one figure per case plus a
combined figure. Case selection is automatic: full fixes, partial fixes,
failures, and regressions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
NAMES = ("nothing", "single peak", "double peaks", "sag")


def load_shapes(steer_dir: Path, name: str) -> list[list[str]]:
    preds = json.loads((steer_dir / f"{name}_predictions.json").read_text())
    return [[p["shape"] for p in curve] for curve in preds]


def short_caption(cap: str, limit: int = 120) -> str:
    text = str(cap)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def draw_panel(ax, curve, shapes, truth, title, stage_edges=(43, 86)):
    ax.plot(curve, color="#1f4e79", lw=1.3)
    for edge in stage_edges:
        ax.axvline(edge, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_xticks([0, 43, 86, 128])
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    labels = []
    ok = 0
    for pred, true in zip(shapes, truth):
        mark = "v" if pred == true else "x"
        if pred == true:
            ok += 1
        labels.append(f"{'OK' if pred == true else pred.upper()}*")
    ax.set_xlabel(f"[{' | '.join(labels)}]  {ok}/3", fontsize=7.5,
                  color="#1e7d32" if ok == 3 else "#c62828")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/etth2_steering_cases")
    parser.add_argument("--n-per-case", type=int, default=2)
    args = parser.parse_args()
    steer = args.steer_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "datasets/etth2_2class/etth2_2class_dataset"

    summary = json.loads((steer / "summary.json").read_text())
    indices = np.asarray(summary["indices"])
    attrs = np.load(data / "test_attrs_idx.npy")[indices]
    truth_shapes = [[NAMES[int(a)] for a in row] for row in attrs]
    truth_curve = np.load(data / "test_ts.npy")[indices, :, 0]
    caps = np.load(data / "test_text_caps.npy", allow_pickle=True).reshape(-1)[indices]

    variants = ["pure", "onp_applied_c2_s1", "onp_applied_c4_s2", "onp_grad_c4_s2"]
    present = [v for v in variants if (steer / f"{v}.npy").exists()]
    curves = {v: np.load(steer / f"{v}.npy") for v in present}
    shapes = {v: load_shapes(steer, v) for v in present}

    def ok_of(variant: str, i: int) -> int:
        return sum(s == t for s, t in zip(shapes[variant][i], truth_shapes[i]))

    # Find case indices automatically.
    def pick(predicate, exclude):
        picked = []
        for i in range(len(indices)):
            if i in exclude or not predicate(i):
                continue
            picked.append(i)
            if len(picked) >= args.n_per_case:
                break
        return picked

    used: set[int] = set()
    fixed = pick(lambda i: ok_of("pure", i) < 3 and ok_of("onp_applied_c2_s1", i) == 3
                 and ok_of("onp_applied_c4_s2", i) >= ok_of("pure", i), used)
    used.update(fixed)
    improved = pick(lambda i: 0 < ok_of("onp_applied_c2_s1", i) - ok_of("pure", i) < 3
                    and ok_of("pure", i) < 3, used)
    used.update(improved)
    failed = pick(lambda i: ok_of("pure", i) == 0 and ok_of("onp_applied_c2_s1", i) == 0
                  and ok_of("onp_applied_c4_s2", i) <= 1, used)
    used.update(failed)
    broken = pick(lambda i: ok_of("pure", i) == 3 and ok_of("onp_applied_c2_s1", i) < 3, used)
    used.update(broken)

    groups = [
        ("Steering fixed all 3 stages", fixed, "g"),
        ("Steering improved (partial)", improved, "b"),
        ("Both wrong (steering failed)", failed, "r"),
        ("Steering broke correct pure", broken, "m"),
    ]
    rows = [i for group in groups for i in group[1]]
    if not rows:
        raise ValueError("no cases found; check variant files and predicate logic")
    n_cols = 1 + len(present)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.4 * n_cols, 2.6 * len(rows)),
                             squeeze=False)
    for row, i in enumerate(rows):
        case = truth_shapes[i]
        axes[row, 0].plot(truth_curve[i], color="#2e7d32", lw=1.4)
        for edge in (43, 86):
            axes[row, 0].axvline(edge, color="gray", ls="--", lw=0.7, alpha=0.6)
        axes[row, 0].set_xticks([0, 43, 86, 128])
        axes[row, 0].tick_params(labelsize=7)
        axes[row, 0].set_title("Ground truth", fontsize=9, fontweight="bold", pad=3)
        axes[row, 0].set_xlabel(" | ".join(case), fontsize=7.5, color="#1e7d32")
        for col, variant in enumerate(present, start=1):
            label = f"{variant} ({ok_of(variant, i)}/3)"
            color = "#1e7d32" if ok_of(variant, i) == 3 else "#c62828"
            draw_panel(axes[row, col], curves[variant][i], shapes[variant][i],
                       truth_shapes[i], label)
            axes[row, col].xaxis.label.set_color(color)
        group_name = next(name for name, ids, _ in groups if i in ids)
        caption = short_caption(caps[i])
        fig.text(0.01, 0.995 - row / max(len(rows), 1) * 0.985,
                 f"#{i} [{group_name}] {caption}", fontsize=6.5, color="gray")
    fig.suptitle(f"etth2_2class steering cases (seed42 on-policy, full test idx {indices[0]}...)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_name = f"{steer.parent.parent.name}_{steer.name}_cases.png"
    fig.savefig(out / out_name, dpi=130)
    print(f"saved {out / out_name}")
    for group_name, ids, _ in groups:
        print(f"{group_name}: indices {ids}")
    plt.close(fig)


if __name__ == "__main__":
    main()
