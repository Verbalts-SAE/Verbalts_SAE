"""Manual case-specific CPU preview; reuse validated residuals without modification."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parent = ROOT / "outputs/cpu_local_morph_preview"
    output = parent / "case_specific"
    metadata = json.loads((parent / "manifest.json").read_text())
    strong_meta = json.loads((parent / "stronger_normalized_manifest.json").read_text())
    assert metadata["cases"] == strong_meta["cases"]
    assert metadata["config"] == strong_meta["config"]
    source = Path(metadata["source"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == metadata["source_sha256"] == strong_meta["source_sha256"]
    with np.load(parent / "preview_arrays.npz", allow_pickle=False) as original:
        ids = original["source_rows"]
        bases = original["backgrounds"]
        residuals = original["residuals"]
        original_samples = original["samples"]
    with np.load(parent / "stronger_normalized_preview_arrays.npz", allow_pickle=False) as strong:
        np.testing.assert_array_equal(ids, strong["source_rows"])
        np.testing.assert_array_equal(residuals, strong["residuals"])
        stronger = strong["backgrounds"]
        stronger_samples = strong["samples"]
    assert bases.shape == residuals.shape == (12, 128)
    np.testing.assert_array_equal(ids[:4], np.arange(4))
    np.testing.assert_array_equal(ids, [r["source_row"] for r in metadata["cases"]])
    np.testing.assert_array_equal(bases, np.load(source, allow_pickle=False)[ids])
    spans = np.ptp(bases, axis=1, keepdims=True)
    tolerance = 1e-12 * np.maximum(1, np.abs(bases).max(axis=1, keepdims=True))
    expected = np.zeros_like(bases)
    valid = (spans > tolerance)[:, 0]
    expected[valid] = ((bases[valid] - bases[valid].min(axis=1, keepdims=True))
                       / spans[valid] * 0.5 - 0.25)
    np.testing.assert_allclose(stronger, expected, atol=1e-14, rtol=0)
    backgrounds = bases.copy()
    backgrounds[4:] = stronger[4:]
    samples = backgrounds + residuals
    np.testing.assert_array_equal(samples[:4], original_samples[:4])
    np.testing.assert_array_equal(samples[4:], stronger_samples[4:])
    np.testing.assert_allclose(samples - backgrounds, residuals, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(samples[residuals == 0], backgrounds[residuals == 0])
    assert np.isfinite(samples).all()
    for record, residual in zip(metadata["cases"], residuals):
        start, stop = record["start"], record["stop_exclusive"]
        assert not residual[:start].any() and not residual[stop:].any()
        if record["injected_kind"] == "nothing":
            assert not residual.any()

    output.mkdir(parents=True, exist_ok=True)

    def draw(ax, background, i, label):
        record = metadata["cases"][i]
        ax.plot(background, color="#3274a1", lw=1.5, label="Background")
        ax.plot(background + residuals[i], color="#e1812c", lw=1.25,
                label="Background + local feature")
        if record["injected_kind"] != "nothing":
            ax.axvspan(record["start"], record["stop_exclusive"] - 1,
                       color="#e1812c", alpha=0.09)
        ax.set(title=f"Row {ids[i]} | {record['injected_kind']} | {label}",
               xlim=(0, 127), xlabel="Sample index", ylabel="Value")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")

    for group, positions in (("first_4", range(4)), ("random_8", range(4, 12))):
        label = "Unchanged background" if group == "first_4" else "Background min-max [-0.25, 0.25]"
        fig, axes = plt.subplots(len(positions) // 2, 2,
                                 figsize=(15, 3.5 * (len(positions) // 2)), squeeze=False)
        for ax, i in zip(axes.flat, positions):
            draw(ax, backgrounds[i], i, label)
        fig.suptitle("Manual case-specific preview | unchanged residuals, A=1.2\n"
                     "No smoothing/filtering or post-normalization | independent y-scales")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output / f"{group}_overlay.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(len(positions), 2, figsize=(15, 3 * len(positions)),
                                 squeeze=False, sharey="row")
        for pair, i in zip(axes, positions):
            draw(pair[0], bases[i], i, "Original background")
            draw(pair[1], backgrounds[i], i, label)
        fig.suptitle("Original vs manual case-specific compression | shared y-limits per row\n"
                     "Identical residuals and locations; no smoothing/filtering")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output / f"{group}_comparison.png", dpi=150)
        plt.close(fig)

    np.savez(output / "preview_arrays.npz", source_rows=ids, original_backgrounds=bases,
             backgrounds=backgrounds, residuals=residuals, samples=samples)
    with np.load(output / "preview_arrays.npz", allow_pickle=False) as saved:
        for key, value in dict(source_rows=ids, backgrounds=backgrounds,
                               residuals=residuals, samples=samples).items():
            np.testing.assert_array_equal(saved[key], value)
    metadata.pop("strength_comparison", None)
    metadata["background_transform"] = "Manual sample assignment: first four unchanged; random eight min-max [-0.25, 0.25] before injection"
    metadata["constant_row_policy"] = strong_meta["constant_row_policy"]
    metadata["arrays_file"] = "preview_arrays.npz"
    metadata["caveats"].append("Manual policy for these 12 samples only; not class-based or automatically generalized")
    for i, record in enumerate(metadata["cases"]):
        record["background_transform"] = "unchanged" if i < 4 else "min-max [-0.25, 0.25]"
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(status="PASS", cases=12, first_four_equal_original=True,
                  random_eight_equal_stronger=True, residuals_identical=True,
                  injection_support_unchanged=True, saved_array_roundtrip=True,
                  source_sha256_unchanged=True)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()