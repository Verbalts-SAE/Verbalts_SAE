"""Build four-stage weather morphology data from disjoint raw time intervals."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_weather_morph_preview import BOUNDS, NAMES, ROOT, build, plt

CHANNELS = ("temperature", "wind_speed", "air_pressure")
COLUMNS = ("T (degC)", "wv (m/s)", "p (mbar)")
SOURCE = Path("/storage/group/renkan/lansc/gents_bench/contsg_repo/datasets/raw/Weather-Captioned/time_series/weather_large.parquet")


def select_scale(raw, channel):
    span = float(np.ptp(raw))
    steps = np.abs(np.diff(raw))
    step_ratio = float(steps.max() / span) if span else 0.0
    variation_ratio = float(steps.sum() / span) if span else 0.0
    gentle = step_ratio <= 0.25 and variation_ratio <= 2.5
    scale = 1.0 if channel == "air_pressure" else (0.3 if gentle else 0.1)
    return scale, {"max_step_ratio": step_ratio, "total_variation_ratio": variation_ratio}


def source_intervals(size):
    """Chronological 80/10/10 partitions with 36-row boundary embargoes."""
    a, b = int(size * 0.8), int(size * 0.9)
    return {"train": (0, a), "valid": (a + 36, b), "test": (b + 36, size)}


def prepare_source(frame):
    frame = frame.copy()
    frame["original_row"] = np.arange(len(frame))
    frame["timestamp"] = pd.to_datetime(frame["Date Time"], format="%d.%m.%Y %H:%M:%S")
    valid = frame["timestamp"].notna() & ~frame["timestamp"].duplicated(keep=False)
    return frame.loc[valid].sort_values("timestamp").reset_index(drop=True)


def make_backgrounds(values, times, interval, count, seed):
    if count <= 0:
        raise ValueError("background count must be positive")
    lo, hi = interval
    # Reserve whole 36-row blocks, including for the short fourth stage.
    candidates = []
    for start in range(lo, hi - 35, 36):
        block = values[start:start + 36]
        regular = np.all(np.diff(times[start:start + 36]) == np.timedelta64(10, "m"))
        physical = (block[:, 1] >= 0).all() and (block[:, 2] > 0).all()
        if np.isfinite(block).all() and physical and regular:
            candidates.append(start)
    if len(candidates) < count * 4:
        raise ValueError(f"Need {count * 4} disjoint source blocks, have {len(candidates)}")
    rng = np.random.default_rng(seed)
    starts = rng.choice(candidates, size=(count, 4), replace=False)
    backgrounds, records = [], []
    for background_id, row in enumerate(starts):
        # Every background includes all three channels; the fourth is random.
        channels = rng.permutation([0, 1, 2, int(rng.integers(3))])
        pieces, segments = [], []
        for start, channel, (out_start, out_stop) in zip(row, channels, BOUNDS):
            start, channel = int(start), int(channel)
            stop = start + out_stop - out_start
            raw = values[start:stop, channel].astype(np.float64)
            scale, diagnostics = select_scale(raw, CHANNELS[channel])
            anchor = float(pieces[-1][-1]) if pieces else 0.0
            piece = (raw - raw[0]) * scale + anchor
            np.testing.assert_allclose(np.diff(piece), np.diff(raw) * scale, atol=1e-12)
            pieces.append(piece)
            segments.append({"source_start": start, "source_stop": stop,
                             "reserved_stop": start + 36,
                             "time_start": str(times[start]), "time_end": str(times[stop - 1]),
                             "channel": CHANNELS[channel], "source_column": COLUMNS[channel],
                             "output_start": out_start, "output_stop": out_stop,
                             "scale": scale, "anchor": anchor, **diagnostics})
        backgrounds.append(np.concatenate(pieces))
        records.append({"background_id": background_id, "segments": segments})
    backgrounds = np.stack(backgrounds)
    if len(np.unique(backgrounds, axis=0)) != count:
        raise ValueError("Duplicate constructed backgrounds; choose another seed")
    return backgrounds, records, len(candidates)


def save_split(out, split, backgrounds, provenance, amplitude):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "backgrounds.npy", backgrounds)
    (out / "background_records.json").write_text(json.dumps(provenance, indent=2) + "\n")
    combinations = list(product(NAMES, repeat=4))
    isolated = [tuple(name if i == stage else "nothing" for i in range(4))
                for stage in range(4) for name in NAMES]
    shapes = {}
    for prefix, combinations_ in (("combined", combinations), ("isolated", isolated)):
        curves, residuals, records = build(backgrounds, combinations_, amplitude)
        bases = backgrounds[[r["background_id"] for r in records]]
        assert np.isfinite(curves).all()
        np.testing.assert_allclose(curves - residuals, bases, atol=1e-12)
        for start, stop in BOUNDS:
            np.testing.assert_array_equal(residuals[:, [start, stop - 1]], 0)
        for boundary in (36, 72, 108):
            np.testing.assert_array_equal(curves[:, boundary], curves[:, boundary - 1])
        labels = np.array([[('nothing', *NAMES).index(n) for n in r['morphologies']]
                           for r in records], dtype=np.int64)
        assert np.all(np.unique(labels, axis=0, return_counts=True)[1] == len(backgrounds))
        for record in records:
            record["split"] = split
        np.save(out / f"{prefix}_samples.npy", curves[..., None])
        np.save(out / f"{prefix}_residuals.npy", residuals[..., None])
        np.save(out / f"{prefix}_labels.npy", labels)
        (out / f"{prefix}_records.json").write_text(json.dumps(records, indent=2) + "\n")
        shapes[prefix] = list(curves[..., None].shape)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    preview, _, _ = build(backgrounds[:6], [NAMES + ("single_peak",)], amplitude)
    for i, ax in enumerate(axes.flat):
        if i >= len(preview):
            ax.set_visible(False)
            continue
        ax.plot(backgrounds[i], "--", color="0.5", label="Weather background")
        ax.plot(preview[i], label="Four-stage injection")
        for start, _ in BOUNDS[1:]:
            ax.axvline(start - 0.5, color="0.7")
        ax.set_title(f"{split}: background {i}")
        ax.legend(fontsize=8)
    fig.savefig(out / "overview.png", dpi=120)
    plt.close(fig)
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/weather_four_stage_morph_40k")
    parser.add_argument("--background-counts", type=int, nargs=3, default=(400, 50, 50),
                        metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--target-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--amplitude", type=float, default=1.75)
    args = parser.parse_args()
    if min(args.background_counts) <= 0 or sum(args.background_counts) * 81 < args.target_samples:
        parser.error("Positive background counts must provide at least target-samples combinations")
    if not np.isfinite(args.amplitude) or args.amplitude <= 0:
        parser.error("amplitude must be finite and positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    frame = pd.read_parquet(args.source, columns=["Date Time", *COLUMNS])
    original_size = len(frame)
    frame = prepare_source(frame)
    times = frame["timestamp"].to_numpy()
    if np.isnat(times).any() or not np.all(np.diff(times) > np.timedelta64(0, "s")):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    values = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    intervals = source_intervals(len(frame))
    prepared = {}
    for index, (split, interval) in enumerate(intervals.items()):
        result = make_backgrounds(values, times, interval, args.background_counts[index], args.seed + index)
        repeated = make_backgrounds(values, times, interval, args.background_counts[index], args.seed + index)
        np.testing.assert_array_equal(result[0], repeated[0])
        assert result[1] == repeated[1]
        for record in result[1]:
            for segment in record["segments"]:
                segment["original_source_rows"] = frame["original_row"].iloc[
                    segment["source_start"]:segment["source_stop"]].tolist()
        prepared[split] = result
    all_backgrounds = np.concatenate([result[0] for result in prepared.values()])
    assert len(np.unique(all_backgrounds, axis=0)) == len(all_backgrounds)
    metadata = {"source": str(args.source.resolve()), "source_rows": len(frame),
                "original_source_rows": original_size, "excluded_timestamp_rows": original_size - len(frame),
                "source_preparation": "Sort by timestamp; exclude all duplicate/NaT timestamp rows; retain original row IDs; no interpolation",
                "seed": args.seed, "amplitude": args.amplitude,
                "stage_bounds": BOUNDS, "label_names": ["nothing", *NAMES],
                "channels": dict(zip(CHANNELS, COLUMNS)),
                "scale_policy": {"temperature_wind_gentle": 0.3, "temperature_wind_other": 0.1,
                                 "pressure": 1.0, "max_step_ratio": 0.25, "max_total_variation_ratio": 2.5},
                "normalization": None, "smoothing": None, "filter": None,
                "split_policy": "chronological 80/10/10; 36-row embargo; disjoint 36-row blocks without replacement",
                "caption_format": "Four numbered stages; not compatible with legacy three-stage parser",
                "splits": {}}
    for split, (backgrounds, records, available) in prepared.items():
        shapes = save_split(args.output_dir / split, split, backgrounds, records, args.amplitude)
        metadata["splits"][split] = {"source_interval": intervals[split],
                                    "available_blocks": available, "background_count": len(backgrounds), **shapes}
        print(f"PASS {split}: {shapes}; reproducible backgrounds, balanced labels, finite, continuous", flush=True)
    metadata["combined_total"] = sum(s["combined"][0] for s in metadata["splits"].values())
    metadata["isolated_total"] = sum(s["isolated"][0] for s in metadata["splits"].values())
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()