"""Four-stage morphology pilot on approved, unmodified weather backgrounds."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contsg.data.datasets.semisynth_morph import InjectionConfig, morphology_residual

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = ((0, 36), (36, 72), (72, 108), (108, 128))
NAMES = ("single_peak", "double_peaks", "sag")


def inject_weather(background, labels, amplitude=1.75):
    background = np.asarray(background, dtype=np.float64)
    if background.shape != (128,) or not np.isfinite(background).all():
        raise ValueError("background must be finite with shape (128,)")
    if len(labels) != 4:
        raise ValueError("exactly four labels are required")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    residual = np.zeros(128)
    for label, (start, stop) in zip(labels, BOUNDS):
        residual[start:stop] = morphology_residual(
            label, stop - start, InjectionConfig(amplitude=amplitude))
    return background + residual, residual


def build(backgrounds, combinations, amplitude):
    curves, residuals, records = [], [], []
    for background_id, background in enumerate(backgrounds):
        for labels in combinations:
            curve, residual = inject_weather(background, labels, amplitude)
            curves.append(curve)
            residuals.append(residual)
            records.append({"sample_id": len(records), "background_id": background_id,
                            "morphologies": list(labels),
                            "caption": " ".join(
                                f"Stage {stage} has {label.replace('_', ' ')}."
                                for stage, label in enumerate(labels, 1))})
    return np.stack(curves), np.stack(residuals), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-dir", type=Path,
                        default=ROOT / "outputs/weather_univariate_preview/normalized_concat_128")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/weather_four_stage_morph_pilot")
    parser.add_argument("--amplitude", type=float, default=1.75)
    args = parser.parse_args()
    backgrounds = np.load(args.background_dir / "samples.npy", allow_pickle=False)
    provenance = json.loads((args.background_dir / "metadata.json").read_text())
    assert backgrounds.shape == (6, 128)
    assert provenance["construction"] == [36, 36, 36, 20]
    for record in provenance["records"]:
        assert [(s["output_start"], s["output_stop"]) for s in record["segments"]] == list(BOUNDS)
    combinations = list(product(NAMES, repeat=4))
    isolated = [tuple(name if i == stage else "nothing" for i in range(4))
                for stage in range(4) for name in NAMES]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "backgrounds.npy", backgrounds)
    for prefix, labels in (("combined", combinations), ("isolated", isolated)):
        curves, residuals, records = build(backgrounds, labels, args.amplitude)
        np.testing.assert_array_equal(curves, build(backgrounds, labels, args.amplitude)[0])
        assert np.isfinite(curves).all()
        bases = backgrounds[[r["background_id"] for r in records]]
        np.testing.assert_allclose(curves - residuals, bases, atol=1e-12)
        for start, stop in BOUNDS:
            np.testing.assert_array_equal(residuals[:, [start, stop - 1]], 0)
        for boundary in (36, 72, 108):
            np.testing.assert_array_equal(curves[:, boundary], curves[:, boundary - 1])
        np.save(out / f"{prefix}_samples.npy", curves[..., None])
        np.save(out / f"{prefix}_residuals.npy", residuals[..., None])
        np.save(out / f"{prefix}_labels.npy",
                [[("nothing", *NAMES).index(n) for n in r["morphologies"]] for r in records])
        (out / f"{prefix}_records.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"PASS {prefix}: {curves.shape}, deterministic, finite, unchanged backgrounds, continuous joins")

    for background_id, background in enumerate(backgrounds):
        fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True,
                                 constrained_layout=True)
        for row, name in enumerate(NAMES):
            for stage in range(4):
                labels = tuple(name if i == stage else "nothing" for i in range(4))
                curve, _ = inject_weather(background, labels, args.amplitude)
                ax = axes[row, stage]
                ax.plot(background, "--", color="0.55", label="Approved background")
                ax.plot(curve, color="tab:blue", label="Background + morphology")
                start, stop = BOUNDS[stage]
                ax.axvspan(start, stop - 1, color="tab:orange", alpha=0.10)
                for boundary in (36, 72, 108):
                    ax.axvline(boundary - 0.5, color="0.7", linewidth=0.6)
                ax.set(title=f"Stage {stage + 1}: {name}", xlim=(0, 127))
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Weather background {background_id}: isolated morphology; amplitude={args.amplitude}")
        fig.savefig(out / f"background_{background_id}_isolated.png", dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    for i, (ax, background) in enumerate(zip(axes.flat, backgrounds)):
        labels = tuple(NAMES[(stage + i) % 3] for stage in range(4))
        curve, _ = inject_weather(background, labels, args.amplitude)
        ax.plot(background, "--", color="0.55", label="Approved background")
        ax.plot(curve, label="Four-stage injection")
        for label, (start, stop) in zip(labels, BOUNDS):
            ax.axvline(start - 0.5, color="0.7", linewidth=0.6)
            ax.text((start + stop - 1) / 2, 0.98, label, ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set(title=f"Background {i}", xlim=(0, 127))
        ax.margins(y=0.25)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Weather: four-stage morphology combinations (no additional normalization)")
    fig.savefig(out / "overview.png", dpi=140)
    plt.close(fig)
    (out / "metadata.json").write_text(json.dumps({
        "background_dir": str(args.background_dir.resolve()), "background_metadata": provenance,
        "stage_bounds": BOUNDS, "label_names": ["nothing", *NAMES],
        "aliases": {"peak": "single_peak", "double_peak": "double_peaks"},
        "injection_config": vars(InjectionConfig(amplitude=args.amplitude)),
        "combined_shape": [486, 128, 1], "isolated_shape": [72, 128, 1],
        "transform": "approved_background + endpoint_zero_morphology_residual",
        "split": None, "purpose": "Visual pilot, not independent train/test data",
        "caption_format": "Four numbered stages; not the legacy three-stage parser format"
    }, indent=2) + "\n")
    print(out.resolve())


if __name__ == "__main__":
    main()