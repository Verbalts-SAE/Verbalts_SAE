"""Isolated background-only median-trend pilot on saved adaptive-v1 arrays."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def denoise_background(background, detail_retention=0.25, window=5):
    """Blend an edge-padded median trend with detail; preserve the input mean."""
    x = np.asarray(background, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional background of length >= 2")
    if not isinstance(window, int) or window < 1 or window % 2 != 1 or window > x.size:
        raise ValueError("Window must be an odd positive integer no longer than the input")
    if not np.isfinite(detail_retention) or not 0 <= detail_retention <= 1:
        raise ValueError("Detail retention must lie in [0, 1]")
    padded = np.pad(x, window // 2, mode="edge")
    trend = np.array([np.median(padded[i:i + window]) for i in range(x.size)])
    y = x + (1 - detail_retention) * (trend - x)
    y += x.mean() - y.mean()
    return y


def diagnostics(x):
    return dict(span=float(np.ptp(x)), total_variation=float(np.abs(np.diff(x)).sum()),
                jump_q90=float(np.quantile(np.abs(np.diff(x)), .9)))


def main():
    parent = ROOT / "outputs/cpu_local_morph_preview"
    previous = parent / "adaptive_v1"
    output = parent / "adaptive_denoised_v1"
    metadata = json.loads((previous / "manifest.json").read_text())
    source = Path(metadata["source"])
    tracked = [source] + [p for p in parent.rglob("*")
                          if p.is_file() and output not in p.parents]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    assert hashes[source] == metadata["source_sha256"]
    assert metadata["config"]["amplitude"] == 1.2
    with np.load(previous / "preview_arrays.npz", allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    ids, backgrounds, residuals = (old[k] for k in ("source_rows", "backgrounds", "residuals"))
    with np.load(parent / "preview_arrays.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(residuals, z["residuals"])
        np.testing.assert_array_equal(ids, z["source_rows"])
    original_metadata = json.loads((parent / "manifest.json").read_text())
    for case, original in zip(metadata["cases"], original_metadata["cases"]):
        for key in ("source_row", "injected_kind", "start", "stop_exclusive"):
            assert case[key] == original[key]
    variants = {"Adaptive v1": backgrounds}
    arrays = dict(source_rows=ids, original_backgrounds=old["original_backgrounds"],
                  adaptive_backgrounds=backgrounds, residuals=residuals)
    for name, retention, key in (("Light: detail 50%", .5, "light"),
                                 ("Moderate: detail 25%", .25, "moderate")):
        result = np.array([denoise_background(b, retention) for b in backgrounds])
        variants[name] = result
        arrays[key + "_backgrounds"] = result
        arrays[key + "_samples"] = result + residuals
        np.testing.assert_allclose(result.mean(axis=1), backgrounds.mean(axis=1), atol=1e-14)
        np.testing.assert_allclose(arrays[key + "_samples"] - result, residuals, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(arrays[key + "_samples"][residuals == 0], result[residuals == 0])
        assert np.isfinite(result).all()
    output.mkdir(parents=True, exist_ok=True)

    def plot_group(indices, filename, background_only=False):
        fig, axes = plt.subplots(len(indices), 3, figsize=(18, 2.7 * len(indices)),
                                 squeeze=False, sharey="row")
        for row, i in zip(axes, indices):
            case = metadata["cases"][i]
            for ax, (label, values) in zip(row, variants.items()):
                ax.plot(values[i], lw=1.2, label="Background")
                if not background_only:
                    ax.plot(values[i] + residuals[i], lw=1.1, label="Background + residual")
                if case["injected_kind"] != "nothing":
                    ax.axvspan(case["start"], case["stop_exclusive"] - 1, alpha=.08, color="orange")
                ax.set(title=f"Row {ids[i]} | {case['injected_kind']} | {label}",
                       xlim=(0, backgrounds.shape[1] - 1))
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
        fig.suptitle("Shared y-limits per row | background-only filtering | unchanged residuals")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output / filename, dpi=120)
        plt.close(fig)

    plot_group(list(range(4)), "first_4_comparison.png")
    plot_group(list(range(4, len(ids))), "random_8_comparison.png")
    plot_group([i for i, c in enumerate(metadata["cases"])
                if c["injected_kind"] == "nothing" and i >= 4], "nothing_detail.png", True)
    for i, case in enumerate(metadata["cases"]):
        case["denoising_metrics"] = {name: diagnostics(values[i]) for name, values in variants.items()}
    metadata["background_transform"] = "adaptive_v1 followed by median5 trend/detail blending, mean preserved"
    metadata["denoising_rule"] = dict(window=5, padding="edge", detail_retentions=[.5, .25],
                                      formula="y=x+(1-retention)*(median5(x)-x); y+=mean(x)-mean(y)",
                                      class_dependent=False)
    metadata["caveats"].append("Median trend is a heuristic, not a ground-truth noise decomposition; narrow natural features may be attenuated. No training integration.")
    np.savez(output / "preview_arrays.npz", **arrays)
    with np.load(output / "preview_arrays.npz", allow_pickle=False) as z:
        for key, value in arrays.items():
            np.testing.assert_array_equal(z[key], value)
    for p, digest in hashes.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, str(p)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = dict(status="PASS", cases=len(ids), residuals_unchanged=True, positions_unchanged=True,
                  amplitude_configuration=1.2, source_and_previous_previews_unchanged=True,
                  saved_array_roundtrip=True, mean_preserved=True)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    for c in metadata["cases"]:
        if c["injected_kind"] == "nothing":
            print(c["source_row"], c["denoising_metrics"], flush=True)


if __name__ == "__main__":
    main()