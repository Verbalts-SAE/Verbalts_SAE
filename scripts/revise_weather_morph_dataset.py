"""Reduce saved weather morphology amplitude and add reproducible observation noise."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.generate_weather_morph_preview import ROOT, BOUNDS, plt


def revise_arrays(backgrounds, residuals, records, ratio, noise_std, seed):
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be finite and positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    bases = backgrounds[[r["background_id"] for r in records]]
    if bases.shape != residuals.shape or not np.isfinite(bases).all() or not np.isfinite(residuals).all():
        raise ValueError("backgrounds and residuals must be finite with matching shapes")
    morphology = residuals * ratio
    noise = np.random.default_rng(seed).normal(0, noise_std, residuals.shape)
    total = morphology + noise
    return bases + total, total, morphology, noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/weather_four_stage_morph_40k")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/weather_four_stage_morph_40k_soft_noise")
    parser.add_argument("--amplitude", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if not np.isfinite(args.amplitude) or args.amplitude <= 0 or not np.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("amplitude must be positive and noise-std nonnegative, both finite")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    metadata = json.loads((args.source_dir / "metadata.json").read_text())
    if metadata.get("observation_noise"):
        parser.error("Source must be a clean dataset without added observation noise")
    old_amplitude = metadata["amplitude"]
    ratio = args.amplitude / old_amplitude
    args.output_dir.mkdir(parents=True)
    for split_index, split in enumerate(("train", "valid", "test")):
        source, out = args.source_dir / split, args.output_dir / split
        out.mkdir()
        backgrounds = np.load(source / "backgrounds.npy", allow_pickle=False)
        background_records = json.loads((source / "background_records.json").read_text())
        assert [r["background_id"] for r in background_records] == list(range(len(backgrounds)))
        for filename in ("backgrounds.npy", "background_records.json"):
            shutil.copy2(source / filename, out / filename)
        for prefix_index, prefix in enumerate(("combined", "isolated")):
            records = json.loads((source / f"{prefix}_records.json").read_text())
            residuals = np.load(source / f"{prefix}_residuals.npy", allow_pickle=False)
            bases = backgrounds[[r["background_id"] for r in records]]
            if bases.ndim == 2:
                bases = bases[..., None]
            old = np.load(source / f"{prefix}_samples.npy", allow_pickle=False)
            np.testing.assert_allclose(old, bases + residuals, atol=2e-6)
            seed = args.seed + split_index * 2 + prefix_index
            arrays = revise_arrays(backgrounds[..., None] if backgrounds.ndim == 2 else backgrounds,
                                   residuals, records, ratio, args.noise_std, seed)
            for suffix, array in zip(("samples", "residuals", "morphology_residuals", "noise"), arrays):
                assert np.isfinite(array).all() and array.shape == old.shape
                np.save(out / f"{prefix}_{suffix}.npy", array)
            np.testing.assert_allclose(arrays[0], bases + arrays[2] + arrays[3], atol=1e-12)
            for suffix in ("labels.npy", "records.json"):
                shutil.copy2(source / f"{prefix}_{suffix}", out / f"{prefix}_{suffix}")
            print(f"PASS {split}/{prefix}: {arrays[0].shape}, noise std={arrays[3].std():.5f}", flush=True)
    metadata.update(parent_dataset=str(args.source_dir.resolve()), amplitude=args.amplitude,
                    observation_noise={"distribution": "Gaussian", "mean": 0, "std": args.noise_std,
                                       "seed": args.seed, "seed_policy": "seed + split_index * 2 + prefix_index"},
                    residual_definition="residuals = morphology_residuals + noise; samples = backgrounds + residuals")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = json.loads((args.source_dir / "different_combinations_cases.json").read_text())
    records = json.loads((args.output_dir / "test/combined_records.json").read_text())
    indices = {r["sample_id"]: i for i, r in enumerate(records)}
    old = np.load(args.source_dir / "test/combined_samples.npy")
    new = np.load(args.output_dir / "test/combined_samples.npy")
    backgrounds = np.load(args.output_dir / "test/backgrounds.npy")
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        index = indices[case["sample_id"]]
        assert records[index]["morphologies"] == case["morphologies"]
        ax.plot(backgrounds[case["background_id"]].reshape(-1), "--", color="0.5", label="Background")
        ax.plot(old[index, :, 0], color="tab:orange", alpha=0.5, label=f"Original amplitude {old_amplitude}")
        ax.plot(new[index, :, 0], color="tab:blue", linewidth=1.2,
                label=f"Amplitude {args.amplitude} + noise std {args.noise_std}")
        for label, (start, stop) in zip(case["morphologies"], BOUNDS):
            if start:
                ax.axvline(start, color="0.75", linewidth=0.8)
            ax.text((start + stop - 1) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set(title=f"Sample {case['sample_id']} | background {case['background_id']}", xlim=(0, 127), xlabel="Sample index")
        ax.margins(y=0.25)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Reduced morphology amplitude ({args.amplitude}) with observation noise (std={args.noise_std})")
    fig.savefig(args.output_dir / "different_combinations.png", dpi=140)
    plt.close(fig)
    (args.output_dir / "different_combinations_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()