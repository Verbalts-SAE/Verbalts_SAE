"""Auditable captions for the CPU detail-50% preview, not training thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_weather36_captions import background_caption, features, fit_thresholds
from scripts.preview_cpu_background_denoising import denoise_background
from scripts.rewrite_electricity_v3_global_captions import MORPHOLOGY_PHRASES

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ("slope", "standard_deviation", "mean_abs_change",
                 "cid_ce_normalized", "autocorrelation_lag_1")


def local_caption(case, length, sample_index=0):
    """Describe the injected morphology, without denying natural background events."""
    kind = case["injected_kind"]
    if kind == "nothing":
        return ""
    if kind not in MORPHOLOGY_PHRASES:
        raise ValueError(f"Unknown morphology: {kind}")
    start, stop = case["start"], case["stop_exclusive"]
    if not 0 <= start < stop <= length:
        raise ValueError("Invalid morphology interval")
    stage_index = min(2, int(3 * (start + stop) / (2 * length)))
    stage = ("beginning", "middle", "end")[stage_index]
    phrases = MORPHOLOGY_PHRASES[kind]
    return phrases[(sample_index + stage_index) % len(phrases)].format(stage=stage)


def fit_preview_thresholds(matrix):
    thresholds = fit_thresholds(matrix)
    for index, name in ((3, "complexity"), (4, "persistence")):
        finite = matrix[np.isfinite(matrix[:, index]), index]
        thresholds[name] = np.quantile(finite, [1/3, 2/3]).tolist() if len(finite) else None
    return thresholds


def global_caption(row, thresholds):
    # Keep sign-aware trend thresholds, but use electricity's natural phrasing.
    trend, pattern = background_caption(row, thresholds).split(" overall, with ")
    text = trend.replace("The background", "The series").replace("noticeably", "steeply") + " overall."
    phrases = [pattern.removesuffix(".")]
    for index, key, labels, noun in (
        (3, "complexity", ("low", "moderate", "fairly pronounced"), "structural complexity"),
        (4, "persistence", ("weak", "moderate", "strong"), "short-term persistence"),
    ):
        if np.isfinite(row[index]) and thresholds[key] is not None:
            level = int(np.searchsorted(thresholds[key], row[index], side="left"))
            phrases.append(f"{labels[level]} {noun}")
    if phrases:
        phrases[0] = phrases[0].replace(" variability and ", " variability, ")
        text += " Its global pattern has " + ", ".join(phrases[:-1]) + (
            ", and " if len(phrases) > 1 else "") + phrases[-1] + "."
    return text


def validate_caption(text):
    """Numeric audit data and preprocessing metadata must never enter captions."""
    forbidden = ("%", "tsfresh", "detail", "amplitude", "source_row", "median", "threshold")
    if any(c.isdigit() for c in text) or any(word in text.lower() for word in forbidden):
        raise ValueError("Caption must contain qualitative descriptions only")


def prepare(source, output):
    source, output = Path(source), Path(output)
    paths = [source / "manifest.json", source / "preview_arrays.npz"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata = json.loads(paths[0].read_text())
    if metadata["config"]["amplitude"] != 1.2:
        raise ValueError("Expected amplitude configuration 1.2")
    with np.load(paths[1], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    backgrounds = arrays["light_backgrounds"]
    cases = metadata["cases"]
    if backgrounds.ndim != 2 or len(cases) != len(backgrounds):
        raise ValueError("Case/background mismatch")
    np.testing.assert_array_equal(arrays["source_rows"], [c["source_row"] for c in cases])
    expected = np.array([denoise_background(x, .5) for x in arrays["adaptive_backgrounds"]])
    np.testing.assert_array_equal(backgrounds, expected)
    np.testing.assert_array_equal(arrays["light_samples"], backgrounds + arrays["residuals"])
    for i, case in enumerate(cases):
        if case["injected_kind"] == "nothing":
            np.testing.assert_array_equal(arrays["residuals"][i], np.zeros(backgrounds.shape[1]))
    matrix = features(backgrounds)
    thresholds = fit_preview_thresholds(matrix)
    records = []
    for index, (case, row) in enumerate(zip(cases, matrix)):
        local = local_caption(case, backgrounds.shape[1], index)
        global_text = global_caption(row, thresholds)
        validate_caption(" ".join([local, global_text]))
        records.append(dict(**case, local_caption=local, global_caption=global_text,
                            caption=" ".join(filter(None, [local, global_text])),
                            tsfresh_features={name: float(v) if np.isfinite(v) else None
                                              for name, v in zip(FEATURE_NAMES, row)}))
    report = dict(status="PASS", samples=len(records), detail_retention=.5,
                  amplitude_configuration=1.2, feature_source="light_backgrounds",
                  threshold_scope="preview_only_not_for_training", thresholds=thresholds,
                  source_hashes=hashes, source_unchanged=True,
                  caption_style="synth-u-electricity-qualitative-morphology-first",
                  captions_numeric_free=True,
                  caveats=["Refit thresholds on training backgrounds only after splitting.",
                           "Local captions describe injected residuals, not verified final-curve labels.",
                           "nothing omits the local sentence; natural background morphology may remain.",
                           "Undefined constant-background autocorrelation is omitted, not imputed."])
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "text_caps.npy", np.asarray([r["caption"] for r in records])[:, None])
    np.save(output / "background_features.npy", matrix)
    (output / "captions.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (output / "captions.md").write_text("# CPU detail=50% caption preview\n\n"
        "Preview-only thresholds; not a training dataset.\n\n" + "\n\n".join(
            f"## Row {r['source_row']} — {r['injected_kind']}\n\n{r['caption']}" for r in records) + "\n")
    np.testing.assert_array_equal(np.load(output / "text_caps.npy")[:, 0], [r["caption"] for r in records])
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cpu_local_morph_preview/adaptive_denoised_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cpu_local_morph_preview/captions_detail50_v2")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))