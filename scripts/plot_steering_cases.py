"""Plot representative steering cases and concept-grid cases (no torch import)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
STEER = ROOT / "results/electricity_v3/steering/valid128_synthustyle_1009823"
GRID = ROOT / "results/electricity_v3/visualizations/verbalts_concept_grid_v2"
OUT = ROOT / "results/electricity_v3/visualizations/steering_case_comparison.png"
OUT2 = ROOT / "results/electricity_v3/visualizations/concept_grid_case_zoom.png"

NAMES = ("nothing", "single peak", "double peaks", "sag")


def load_shapes(name):
    preds = json.loads((STEER / f"{name}_predictions.json").read_text())
    return [[p["shape"] for p in c] for c in preds]


def draw_curve(ax, curve, shapes, truth, title, stage_edges=(43, 86)):
    ax.plot(curve, color="#1f4e79", lw=1.4)
    for edge in stage_edges:
        ax.axvline(edge, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xticks([0, 43, 86, 128])
    ax.tick_params(labelsize=8)
    labels = []
    for pred, true in zip(shapes, truth):
        labels.append(f"{pred}" if pred == true else f"{pred}*")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    text = " | ".join(labels)
    ok = sum(p == t for p, t in zip(shapes, truth))
    ax.set_xlabel(f"[{text}]  ({ok}/3)", fontsize=8, color="#1e7d32" if ok == 3 else "#c62828")


def main():
    summary = json.loads((STEER / "summary.json").read_text())
    indices = np.asarray(summary["indices"])
    attrs = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_attrs_idx.npy")[indices]
    truth = [[NAMES[a] for a in row] for row in attrs]
    caps = np.load(
        ROOT / "datasets/electricity_15min_semisynth_morph/valid_text_caps.npy",
        allow_pickle=True,
    ).reshape(-1)[indices]
    gt = np.load(ROOT / "datasets/electricity_15min_semisynth_morph/valid_ts.npy")[indices, :, 0]
    pure = np.load(STEER / "pure.npy")
    dyn16 = np.load(STEER / "dynamic_k16_e5120_g6.npy")
    dense = np.load(STEER / "dense_k0_e5120_g6.npy")
    sp, sd, sdense = load_shapes("pure"), load_shapes("dynamic_k16_e5120_g6"), load_shapes("dense_k0_e5120_g6")

    cases = [
        (50, "Steering fixed it (pure wrong -> dyn correct)"),
        (109, "Steering fixed it (pure wrong -> dyn correct)"),
        (59, "Steering broke it (pure correct -> dyn wrong)"),
        (127, "Steering broke it (pure correct -> dyn wrong)"),
        (1, "Both wrong (middle single peak missed)"),
        (6, "Both wrong (middle single peak missed)"),
    ]
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 2.6 * len(cases)))
    col_titles = ["Ground truth", "Pure VerbalTS", "Dynamic steering (k=16)", "Dense steering (k=0)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)
    for row, (pos, label) in enumerate(cases):
        t = truth[pos]
        axes[row, 0].set_ylabel(f"case {pos}\n(valid {indices[pos]})", fontsize=8, rotation=0, labelpad=42)
        draw_curve(axes[row, 0], gt[pos], t, t, f"truth: {t[0]}/{t[1]}/{t[2]}")
        draw_curve(axes[row, 1], pure[pos], sp[pos], t, "pure")
        draw_curve(axes[row, 2], dyn16[pos], sd[pos], t, "dyn k16 e5120 g6")
        draw_curve(axes[row, 3], dense[pos], sdense[pos], t, "dense k0 e5120 g6")
        fig.text(0.005, 1 - (row + 0.5) / len(cases), label, fontsize=9, va="center", color="#4a148c")
        cap = str(caps[pos])
        fig.text(0.005, 1 - (row + 0.5) / len(cases) - 0.018, cap[:96], fontsize=7, va="center", color="#555555")
    fig.suptitle(
        "Steering case comparison (valid128_synthustyle_1009823) — * marks a stage the CNN predicted wrong",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)

    gs = json.loads((GRID / "visualization_summary.json").read_text())
    rows = gs["rows"]
    pick_concepts = ["beginning_single_peak", "middle_double_peaks", "end_single_peak", "middle_nothing"]
    gen = np.load(GRID / "generated.npy")
    ref = np.load(GRID / "reference.npy")
    fig2, axes2 = plt.subplots(len(pick_concepts), 2, figsize=(12, 2.5 * len(pick_concepts)))
    for row_idx, concept in enumerate(pick_concepts):
        info = next(r for r in rows if r["concept"] == concept)
        targets = [t for t in info["attrs_idx"]]
        truth2 = [NAMES[a] for a in targets]
        preds = info["predicted_shapes"]
        grid_row = rows.index(info)
        for col_idx, (data, kind) in enumerate(((ref, "reference"), (gen, "generated"))):
            ax = axes2[row_idx, col_idx]
            ax.plot(data[grid_row], color="#1f4e79", lw=1.4)
            for edge in (43, 86):
                ax.axvline(edge, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.set_ylim(-4.2, 4.2)
            ax.set_xticks([0, 43, 86, 128])
            ax.tick_params(labelsize=8)
            if kind == "generated":
                labels = [f"{p}" if p == t else f"{p}*" for p, t in zip(preds, truth2)]
                ax.set_xlabel(" | ".join(labels), fontsize=8)
        axes2[row_idx, 0].set_ylabel(
            f"{concept}\n{info['caption'][:60]}", fontsize=7.5, rotation=0, labelpad=60
        )
    axes2[0, 0].set_title("Reference", fontsize=11)
    axes2[0, 1].set_title("Generated (synth-u captions)", fontsize=11)
    fig2.suptitle(
        "Concept-grid zoom: correct / two failure cases / all-nothing (* = wrong stage)",
        fontsize=12, y=1.0,
    )
    fig2.tight_layout()
    fig2.savefig(OUT2, dpi=150, bbox_inches="tight")
    print("saved", OUT2)


if __name__ == "__main__":
    main()
