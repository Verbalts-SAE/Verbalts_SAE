"""Generate single-channel 36-point weather samples with three local stages."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contsg.data.datasets.semisynth_morph import InjectionConfig, make_caption, morphology_residual
from scripts.generate_weather_morph_dataset import CHANNELS, COLUMNS, SOURCE, prepare_source, source_intervals
from scripts.generate_weather_morph_preview import ROOT, NAMES, plt

BOUNDS = ((0, 12), (12, 24), (24, 36))
LABELS = ("nothing", *NAMES)


def make_backgrounds(values, times, interval, count, seed):
    """Use disjoint time blocks and one balanced channel per block; never stitch."""
    lo, hi = interval
    if count <= 0 or not 0 <= lo < hi <= len(values):
        raise ValueError("Invalid count or interval")
    candidates = []
    for start in range(lo, hi - 35, 36):
        block = values[start:start + 36]
        if (np.isfinite(block).all() and (block[:, 1] >= 0).all()
                and (block[:, 2] > 0).all()
                and np.all(np.diff(times[start:start + 36]) == np.timedelta64(10, "m"))):
            candidates.append(start)
    if len(candidates) < count:
        raise ValueError(f"Need {count} disjoint blocks, have {len(candidates)}")
    rng = np.random.default_rng(seed)
    starts = rng.choice(candidates, count, replace=False)
    channels = rng.permutation(np.arange(count) % 3)
    backgrounds, records = [], []
    for i, (start, channel) in enumerate(zip(starts, channels)):
        start, channel = int(start), int(channel)
        raw = values[start:start + 36, channel].astype(np.float64)
        span = float(np.ptp(raw))
        steps = np.abs(np.diff(raw))
        max_step = float(steps.max() / span) if span else 0.0
        variation = float(steps.sum() / span) if span else 0.0
        scale = 1.0 if channel == 2 else (0.4 if max_step <= 0.25 and variation <= 2.5 else 0.2)
        backgrounds.append((raw - raw[0]) * scale)
        records.append({"background_id": i, "channel": CHANNELS[channel],
                        "source_column": COLUMNS[channel], "source_start": start,
                        "source_stop": start + 36, "time_start": str(times[start]),
                        "time_end": str(times[start + 35]), "raw_first": float(raw[0]),
                        "scale": scale, "max_step_ratio": max_step,
                        "total_variation_ratio": variation})
    return np.stack(backgrounds), records, len(candidates)


def inject(background, labels, amplitude=1.2):
    background = np.asarray(background, dtype=np.float64)
    if background.shape != (36,) or not np.isfinite(background).all():
        raise ValueError("background must be finite with shape (36,)")
    if len(labels) != 3 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Require three labels and a finite positive amplitude")
    residual = np.zeros(36)
    for label, (start, stop) in zip(labels, BOUNDS):
        residual[start:stop] = morphology_residual(label, stop - start, InjectionConfig(amplitude=amplitude))
    return background + residual, residual


def save_split(out, backgrounds, provenance, amplitude, noise_std, seed):
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "backgrounds.npy", backgrounds)
    (out / "background_records.json").write_text(json.dumps(provenance, indent=2) + "\n")
    combinations = list(product(NAMES, repeat=3))
    isolated = [tuple(name if i == stage else "nothing" for i in range(3))
                for stage in range(3) for name in NAMES]
    shapes = {}
    for offset, (prefix, patterns) in enumerate((("combined", combinations), ("isolated", isolated))):
        records, morphology = [], []
        for bg_id, background in enumerate(backgrounds):
            for labels in patterns:
                morphology.append(inject(background, labels, amplitude)[1])
                records.append({"sample_id": len(records), "background_id": bg_id,
                                "channel": provenance[bg_id]["channel"],
                                "morphologies": list(labels), "caption": make_caption(labels)})
        morphology = np.stack(morphology)[..., None]
        noise = np.random.default_rng(seed + offset).normal(0, noise_std, morphology.shape)
        residual = morphology + noise
        bases = backgrounds[[r["background_id"] for r in records], :, None]
        samples = bases + residual
        for suffix, array in (("samples", samples), ("residuals", residual),
                              ("morphology_residuals", morphology), ("noise", noise)):
            assert np.isfinite(array).all()
            np.save(out / f"{prefix}_{suffix}.npy", array)
        np.save(out / f"{prefix}_labels.npy", [[LABELS.index(n) for n in r["morphologies"]] for r in records])
        (out / f"{prefix}_records.json").write_text(json.dumps(records, indent=2) + "\n")
        np.testing.assert_allclose(samples, bases + morphology + noise, atol=1e-12)
        shapes[prefix] = list(samples.shape)
    return shapes


def plot_preview(out):
    split = out / "test"
    backgrounds = np.load(split / "backgrounds.npy")
    samples = np.load(split / "combined_samples.npy")
    records = json.loads((split / "background_records.json").read_text())
    combinations = list(product(NAMES, repeat=3))
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for channel, row in enumerate(axes):
        ids = [r["background_id"] for r in records if r["channel"] == CHANNELS[channel]]
        for col, ax in enumerate(row):
            bg = ids[col % len(ids)]
            pattern = (channel * 9 + col * 5) % 27
            ax.plot(backgrounds[bg], "--", color="0.5", label="Single-channel background")
            ax.plot(samples[bg * 27 + pattern, :, 0], label="Morphology + noise")
            for label, (start, stop) in zip(combinations[pattern], BOUNDS):
                ax.text((start + stop - 1) / 2, 0.98, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=9)
            for boundary in (11.5, 23.5):
                ax.axvline(boundary, color="0.7", linewidth=0.8)
            ax.set(title=f"{CHANNELS[channel]} | background {bg}", xlim=(0, 35), xlabel="Sample index")
            ax.margins(y=0.3)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("36 consecutive observations, one channel; beginning / middle / end")
    fig.savefig(out / "overview.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/weather_three_stage_36_morph_40k")
    parser.add_argument("--background-counts", nargs=3, type=int, default=[1200, 150, 150])
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--amplitude", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.08)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; refusing to overwrite")
    if min(args.background_counts) < 3:
        parser.error("At least three backgrounds per split are required")
    if not np.isfinite(args.amplitude) or args.amplitude <= 0 or not np.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("Invalid amplitude or noise standard deviation")
    frame = prepare_source(pd.read_parquet(args.source, columns=["Date Time", *COLUMNS]))
    times = frame["timestamp"].to_numpy()
    values = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    intervals = source_intervals(len(frame))
    metadata = {"source": str(args.source.resolve()), "source_rows": len(frame),
                "sequence_length": 36, "stage_bounds": BOUNDS, "label_names": LABELS,
                "seed": args.seed, "amplitude": args.amplitude, "noise_std": args.noise_std,
                "construction": "One channel, one consecutive 36-row block; no concatenation, padding or resampling",
                "background_transform": "(raw - raw[0]) * scale, once per entire sample",
                "scale_policy": {"temperature_wind_gentle": 0.4, "temperature_wind_other": 0.2,
                                 "pressure": 1.0, "max_step_ratio": 0.25, "max_total_variation_ratio": 2.5},
                "split_policy": "Chronological 80/10/10 with 36-row embargo; disjoint time blocks",
                "noise_seed_policy": "seed + split_index * 2 + prefix_index; iid Gaussian",
                "residual_definition": "samples = background + morphology_residuals + noise; residuals = morphology_residuals + noise",
                "splits": {}}
    for index, (split, interval) in enumerate(intervals.items()):
        backgrounds, records, available = make_backgrounds(values, times, interval, args.background_counts[index], args.seed + index)
        for record in records:
            record["original_source_rows"] = frame["original_row"].iloc[record["source_start"]:record["source_stop"]].tolist()
        shapes = save_split(args.output_dir / split, backgrounds, records, args.amplitude, args.noise_std, args.seed + index * 2)
        metadata["splits"][split] = {"source_interval": interval, "available_blocks": available,
                                      "background_count": len(backgrounds), **shapes}
        print(f"PASS {split}: {shapes}", flush=True)
    metadata["combined_total"] = sum(s["combined"][0] for s in metadata["splits"].values())
    metadata["isolated_total"] = sum(s["isolated"][0] for s in metadata["splits"].values())
    plot_preview(args.output_dir)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()