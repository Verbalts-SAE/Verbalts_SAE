"""Preview fixed local captions plus conservative, background-only trajectory captions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.preview_cpu_background_denoising import denoise_background

ROOT = Path(__file__).resolve().parents[1]
# Pilot heuristics in the existing normalized signal units, not fitted to test data.
# These require training-only calibration before use in a full training dataset.
RULE = dict(bins=16, flat_range=0.18, min_change=0.18,
            relative_change=0.20, min_r2=0.65, min_piecewise_improvement=0.35)
DESCRIPTIONS = {
    "flat": "The background stays roughly level.",
    "rise": "The background generally rises across the sequence.",
    "fall": "The background generally falls across the sequence.",
    "rise_fall": "The background rises and then falls.",
    "fall_rise": "The background falls and then rises.",
    "rise_flat": "The background rises and then levels off.",
    "fall_flat": "The background falls and then levels off.",
    "flat_rise": "The background stays roughly level at first and then rises.",
    "flat_fall": "The background stays roughly level at first and then falls.",
    "irregular": "The background fluctuates without a clear overall direction.",
}


def trajectory(background, rule=None):
    """Fit a line or one continuous hinge to bin medians; never modify the signal."""
    rule = dict(RULE if rule is None else rule)
    x = np.asarray(background, dtype=float)
    if x.ndim != 1 or len(x) < rule["bins"] or not np.isfinite(x).all():
        raise ValueError("Expected a finite 1-D background with at least one point per bin")
    chunks = np.array_split(np.arange(len(x)), rule["bins"])
    times = np.array([indices.mean() for indices in chunks]) / (len(x) - 1)
    coarse = np.array([np.median(x[indices]) for indices in chunks])
    centered = coarse - coarse.mean()
    total = float(centered @ centered)
    span = float(np.ptp(coarse))
    rough_error = np.concatenate([x[indices] - value for indices, value in zip(chunks, coarse)])
    noise = float(1.4826 * np.median(np.abs(rough_error - np.median(rough_error))))
    change_floor = max(rule["min_change"], rule["relative_change"] * span, 2 * noise)

    def fit(knot=None):
        columns = [np.ones_like(times), times]
        if knot is not None:
            columns.append(np.maximum(0, times - knot))
        design = np.column_stack(columns)
        coefficients = np.linalg.lstsq(design, coarse, rcond=None)[0]
        fitted = design @ coefficients
        error = float(np.sum((coarse - fitted) ** 2))
        return coefficients, fitted, error

    line, fitted, line_error = fit()
    candidates = [(k, *fit(k)) for k in times[3:-3]]
    knot, coefficients, piece, piece_error = min(candidates, key=lambda item: item[3])
    improvement = max(0., (line_error - piece_error) / max(line_error, 1e-12))
    line_r2 = 1 - line_error / max(total, 1e-12)
    piece_r2 = 1 - piece_error / max(total, 1e-12)
    first_change = float(coefficients[1] * (knot - times[0]))
    last_change = float((coefficients[1] + coefficients[2]) * (times[-1] - knot))

    def direction(change):
        return "flat" if abs(change) < change_floor else "rise" if change > 0 else "fall"

    directions = (direction(first_change), direction(last_change))
    shape = "irregular"
    model = "line"
    if span <= rule["flat_range"]:
        shape, model = "flat", "constant"
        fitted = np.full_like(coarse, coarse.mean())
    elif (improvement >= rule["min_piecewise_improvement"] and piece_r2 >= rule["min_r2"]
          and directions[0] != directions[1]):
        shape = "_".join(directions)
        model, fitted = "piecewise", piece
    elif line_r2 >= rule["min_r2"] and direction(float(fitted[-1] - fitted[0])) != "flat":
        shape = direction(float(fitted[-1] - fitted[0]))
    return dict(shape=shape, caption=DESCRIPTIONS[shape], model=model,
                review_required=shape == "irregular", bin_times=times.tolist(),
                bin_medians=coarse.tolist(), fit=fitted.tolist(), coarse_span=span,
                change_floor=change_floor, within_bin_noise=noise,
                line_r2=line_r2, piecewise_r2=piece_r2, piecewise_improvement=improvement,
                candidate_knot=float(knot), candidate_changes=[first_change, last_change])


def local_caption(case, length):
    phrases = dict(single_peak="a single peak", double_peaks="double peaks", sag="a sag")
    kind = case["injected_kind"]
    if kind == "nothing":
        return ""
    if kind not in phrases:
        raise ValueError(f"Unknown morphology: {kind}")
    start, stop = case["start"], case["stop_exclusive"]
    if not 0 <= start < stop <= length:
        raise ValueError("Invalid morphology interval")
    stage = ("beginning", "middle", "end")[min(2, int(3 * (start + stop) / (2 * length)))]
    return f"The {stage} part has {phrases[kind]}."


def validate_caption(text):
    forbidden = ("%", "tsfresh", "amplitude", "detail", "complexity", "persistence", "threshold")
    if any(c.isdigit() for c in text) or any(term in text.lower() for term in forbidden):
        raise ValueError("Expected a qualitative visual-shape caption")


def prepare(source, previous, output):
    source, previous, output = map(Path, (source, previous, output))
    paths = [source / "manifest.json", source / "preview_arrays.npz", previous / "captions.json"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    meta = json.loads(paths[0].read_text())
    old = json.loads(paths[2].read_text())
    with np.load(paths[1], allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    bg, residuals = arrays["light_backgrounds"], arrays["residuals"]
    cases = meta["cases"]
    if meta["config"]["amplitude"] != 1.2 or len(cases) != len(bg) or len(old) != len(bg):
        raise ValueError("Unexpected preview configuration or case count")
    np.testing.assert_array_equal(arrays["source_rows"], [c["source_row"] for c in cases])
    np.testing.assert_array_equal(arrays["source_rows"], [c["source_row"] for c in old])
    np.testing.assert_array_equal(bg, [denoise_background(x, .5) for x in arrays["adaptive_backgrounds"]])
    np.testing.assert_array_equal(arrays["light_samples"], bg + residuals)
    records = []
    for i, case in enumerate(cases):
        if case["injected_kind"] == "nothing":
            np.testing.assert_array_equal(residuals[i], np.zeros(bg.shape[1]))
        diagnostic = trajectory(bg[i])
        local = local_caption(case, bg.shape[1])
        caption = " ".join(filter(None, [local, diagnostic["caption"]]))
        validate_caption(caption)
        records.append(dict(source_row=case["source_row"], injected_kind=case["injected_kind"],
                            local_caption=local, global_caption=diagnostic["caption"], caption=caption,
                            previous_caption=old[i]["caption"], trajectory=diagnostic,
                            local_visibility_verified=False))
    output.mkdir(parents=True, exist_ok=False)
    caps = np.asarray([r["caption"] for r in records])[:, None]
    np.save(output / "text_caps.npy", caps)
    (output / "captions.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (output / "captions.md").write_text("# Shape caption preview\n\n"
        "Pilot rules only. Row headings are audit identifiers, not caption text.\n\n" + "\n\n".join(
            f"## Row {r['source_row']} — {r['injected_kind']}\n\n"
            f"**New:** {r['caption']}\n\n**Previous:** {r['previous_caption']}\n\n"
            f"Background review required: {r['trajectory']['review_required']}. "
            "Local visibility on the final curve has not been verified."
            for r in records) + "\n")
    for start in range(0, len(records), 4):
        group = records[start:start + 4]
        fig, axes = plt.subplots(len(group), 2, figsize=(17, 3.6 * len(group)), squeeze=False)
        for offset, (record, row) in enumerate(zip(group, axes)):
            i = start + offset
            diagnostic = record["trajectory"]
            row[0].plot(bg[i], label="Background", lw=1.2)
            row[0].plot(arrays["light_samples"][i], label="Final signal", lw=1)
            times = np.array(diagnostic["bin_times"]) * (bg.shape[1] - 1)
            row[0].plot(times, diagnostic["bin_medians"], "o", ms=3, label="Caption-only coarse view")
            row[0].plot(times, diagnostic["fit"], "--", label="Caption-only fit")
            case = cases[i]
            if case["injected_kind"] != "nothing":
                row[0].axvspan(case["start"], case["stop_exclusive"] - 1, color="orange", alpha=.12)
            row[0].set_title(f"Row {record['source_row']} | {record['injected_kind']} | {diagnostic['shape']}")
            row[0].legend(fontsize=7)
            row[0].grid(alpha=.2)
            row[1].axis("off")
            text = "NEW\n" + textwrap.fill(record["caption"], 83) + "\n\nPREVIOUS\n" + textwrap.fill(record["previous_caption"], 83)
            if diagnostic["review_required"]:
                text += "\n\nREVIEW: no reliable simple background trajectory."
            row[1].text(0, .98, text, va="top", fontsize=10, transform=row[1].transAxes)
        fig.tight_layout()
        fig.savefig(output / f"comparison_{start + 1:02d}_{start + len(group):02d}.png", dpi=120)
        plt.close(fig)
    np.testing.assert_array_equal(np.load(output / "text_caps.npy"), caps)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed: {path}")
    report = dict(status="PASS", samples=len(records), rule=RULE, source_hashes=hashes,
                  source_unchanged=True, caption_only_fitting=True, detail_retention=.5,
                  amplitude_configuration=1.2, numeric_free=True, training_started=False,
                  shape_counts=dict(Counter(r["trajectory"]["shape"] for r in records)),
                  review_rows=[r["source_row"] for r in records if r["trajectory"]["review_required"]],
                  caveats=["Pilot heuristics, not learned or validated training thresholds.",
                           "Coarse medians suppress narrow background events only for description.",
                           "Local labels describe injections; final-curve visibility still needs review.",
                           "One-breakpoint model cannot fully describe multiple broad turns."])
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT / "outputs/cpu_local_morph_preview"
    parser.add_argument("--source", type=Path, default=parent / "adaptive_denoised_v1")
    parser.add_argument("--previous", type=Path, default=parent / "captions_detail50_v2")
    parser.add_argument("--output", type=Path, default=parent / "captions_shape_v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.previous, args.output), indent=2))