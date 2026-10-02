"""Preview additive local morphology on unchanged, already standardized CPU rows.

Run from the repository root with python -m scripts.preview_cpu_local_morph.
This is a visual pilot, not a labeled or split training dataset.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contsg.data.datasets.semisynth_morph import InjectionConfig, morphology_residual


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("single_peak", "double_peaks", "sag", "nothing")
INTERVALS = ((12, 37), (51, 76), (91, 116))


def local_residual(kind, start, stop, config):
    residual = np.zeros(128, dtype=np.float64)
    residual[start:stop] = morphology_residual(kind, stop - start, config)
    return residual


def main():
    source = ROOT / "datasets/CPU/daily128_z.npy"
    output = ROOT / "outputs/cpu_local_morph_preview"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = np.load(source, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 128 or len(data) < 12:
        raise ValueError("Expected at least twelve length-128 CPU rows")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite source data")
    random_ids = np.random.default_rng(42).choice(
        np.arange(4, len(data)), size=8, replace=False
    ).tolist()
    ids = list(range(4)) + random_ids
    bases = data[ids].astype(np.float64)
    config = InjectionConfig(amplitude=1.2, width=0.08, double_separation=0.34)
    records, residuals = [], []
    for i, index in enumerate(ids):
        kind = KINDS[i % len(KINDS)]
        start, stop = INTERVALS[i % len(INTERVALS)]
        residual = local_residual(kind, start, stop, config)
        residuals.append(residual)
        records.append(dict(source_row=index, injected_kind=kind, start=start,
                            stop_exclusive=stop, peak_abs_increment=float(abs(residual).max())))
    residuals = np.asarray(residuals)
    samples = bases + residuals

    # Verify that the pilot never changes background values outside the support.
    np.testing.assert_array_equal(bases, data[ids])
    np.testing.assert_array_equal(samples[residuals == 0], bases[residuals == 0])
    np.testing.assert_allclose(samples - bases, residuals, atol=1e-14, rtol=0)
    assert np.isfinite(samples).all()
    for record, residual in zip(records, residuals):
        start, stop = record["start"], record["stop_exclusive"]
        assert residual[start] == residual[stop - 1] == 0
        assert not residual[:start].any() and not residual[stop:].any()
        if record["injected_kind"] == "nothing":
            assert not residual.any()

    output.mkdir(parents=True, exist_ok=True)

    def draw(ax, base, residual, record, title):
        ax.plot(base, color="#3274a1", lw=1.5, label="Stored z background (unchanged)")
        ax.plot(base + residual, color="#e1812c", lw=1.25, alpha=0.9,
                label="Background + local feature")
        if record["injected_kind"] != "nothing":
            ax.axvspan(record["start"], record["stop_exclusive"] - 1,
                       color="#e1812c", alpha=0.09)
        ax.set(title=title, xlim=(0, 127), xlabel="Sample index (cadence unknown)",
               ylabel="Stored standardized units")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")

    for group, positions in (("first_4", range(4)), ("random_8", range(4, 12))):
        fig, axes = plt.subplots(len(positions) // 2, 2,
                                 figsize=(15, 3.5 * (len(positions) // 2)), squeeze=False)
        for ax, i in zip(axes.flat, positions):
            record = records[i]
            draw(ax, bases[i], residuals[i], record,
                 f"Row {record['source_row']} | injected: {record['injected_kind']}")
        fig.suptitle("CPU local-feature pilot | amplitude parameter 1.2\n"
                     "No new normalization, smoothing, clipping, resampling or noise; nothing = no injection")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output / f"{group}_overlay.png", dpi=150)
        plt.close(fig)

    strengths = (0.6, 1.2, 1.8)
    fig, axes = plt.subplots(4, 3, figsize=(19, 12), squeeze=False)
    for i in range(4):
        record = records[i]
        variants = [local_residual(record["injected_kind"], record["start"],
                                  record["stop_exclusive"], InjectionConfig(amplitude=a))
                    for a in strengths]
        low = min(bases[i].min(), *(float((bases[i] + r).min()) for r in variants)) - 0.3
        high = max(bases[i].max(), *(float((bases[i] + r).max()) for r in variants)) + 0.3
        for ax, amplitude, residual in zip(axes[i], strengths, variants):
            draw(ax, bases[i], residual, record,
                 f"Row {record['source_row']} | {record['injected_kind']} | A={amplitude}")
            ax.set_ylim(low, high)
    fig.suptitle("Same background and location, three injection strengths | shared y-limits per row")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "strength_comparison.png", dpi=150)
    plt.close(fig)

    np.savez(output / "preview_arrays.npz", source_rows=ids, backgrounds=bases,
             residuals=residuals, samples=samples)
    metadata = dict(source=str(source), source_sha256=digest, seed=42,
                    selection="First four plus eight random rows excluding first four; no filtering",
                    background_transform="None: stored rows already have near-zero mean and unit std",
                    config=asdict(config), strength_comparison=list(strengths), cases=records,
                    caveats=["Source preprocessing and cadence remain unverified",
                             "Injected kind is not a verified label for the final curve",
                             "nothing means no injection, not absence of natural morphology",
                             "Preview only; not a train/validation/test split"])
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with np.load(output / "preview_arrays.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["samples"], samples)
        np.testing.assert_array_equal(saved["backgrounds"], data[saved["source_rows"]])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    print(f"PASS: 12 cases; unchanged background, zero outside support, saved-array roundtrip, source SHA256 unchanged.\n{output}")


if __name__ == "__main__":
    main()