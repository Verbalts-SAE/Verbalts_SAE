"""Plot fixed, unselected same-index cases from the two CPU arrays."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "datasets/CPU"
    output = ROOT / "outputs/cpu_case_preview"
    output.mkdir(parents=True, exist_ok=True)
    z = np.load(source / "daily128_z.npy", allow_pickle=False)
    morph = np.load(source / "daily128_morph.npy", allow_pickle=False)
    if z.shape != morph.shape or z.ndim != 2 or len(z) < 12:
        raise ValueError("Expected matching 2D arrays with at least 12 rows")
    if not np.isfinite(z).all() or not np.isfinite(morph).all():
        raise ValueError("Nonfinite values in source arrays")
    seed = 42
    random_ids = np.random.default_rng(seed).choice(
        np.arange(4, len(z)), size=8, replace=False
    ).tolist()
    groups = {"random_8": random_ids, "first_4": list(range(4))}
    x = np.arange(z.shape[1])

    def draw(ax, index, mode):
        if mode in ("z", "overlay"):
            ax.plot(x, z[index], color="#3274a1", lw=1.4, label="daily128_z")
        if mode in ("morph", "overlay"):
            ax.plot(x, morph[index], color="#e1812c", lw=1.6,
                    label="daily128_morph")
        low = min(z[index].min(), morph[index].min())
        high = max(z[index].max(), morph[index].max())
        pad = max(float(high - low) * 0.10, 0.1)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(0, len(x) - 1)
        ax.set_title(f"Case {index} | {mode}", fontsize=10)
        ax.set_xlabel("Sample index (time interval unknown)", fontsize=8)
        ax.set_ylabel("Stored value", fontsize=9)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, loc="best")

    for group, ids in groups.items():
        fig, axes = plt.subplots(len(ids), 3, figsize=(17, 2.5 * len(ids)),
                                 squeeze=False)
        for row, index in enumerate(ids):
            for col, mode in enumerate(("z", "morph", "overlay")):
                draw(axes[row, col], index, mode)
        fig.suptitle("CPU arrays: same row index, unchanged stored values\n"
                     "Matching provenance is not verified; y-limits shared within each row",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(output / f"{group}_comparison.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(len(ids) // 2, 2,
                                 figsize=(14, 3 * (len(ids) // 2)), squeeze=False)
        for ax, index in zip(axes.flat, ids):
            draw(ax, index, "overlay")
        fig.suptitle(f"CPU same-index overlays | {group} | no extra smoothing",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output / f"{group}_overlay.png", dpi=150)
        plt.close(fig)

    (output / "manifest.json").write_text(json.dumps({
        "sources": [str(source / name) for name in
                    ("daily128_z.npy", "daily128_morph.npy")],
        "seed": seed,
        "selection": "First four rows and eight random rows excluding the first four",
        "zero_based_case_ids": groups,
        "transform": "None; original stored values",
        "note": "Same-index comparison does not establish paired provenance",
    }, indent=2) + "\n")
    print(json.dumps(groups), flush=True)
    print(f"Saved plots to {output}", flush=True)


if __name__ == "__main__":
    main()