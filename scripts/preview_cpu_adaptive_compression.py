"""Background-only heuristic pilot; no training integration or signal filtering."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compress_background(background, amplitude=1.2, budget_multiplier=1.0):
    """Positive scalar compression about the mean; never amplify a background."""
    x = np.asarray(background, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional background of length >= 2")
    if amplitude <= 0 or budget_multiplier <= 0:
        raise ValueError("Budgets must be positive")
    span = float(np.ptp(x))
    delta = np.abs(np.diff(x))
    roughness = float(delta.sum() / span) if span > 0 else 0.0
    jump = float(np.quantile(delta, 0.9))
    # Smooth transition: coherent broad excursions retain a larger range budget.
    target = amplitude * budget_multiplier * (
        0.5 + 2.5 * np.exp(-(max(roughness - 4.0, 0.0) / 2.2) ** 2)
    )
    jump_budget = 0.2 * amplitude * budget_multiplier
    scale = min(1.0, target / span if span else 1.0,
                jump_budget / jump if jump else 1.0)
    result = x.copy() if scale == 1 else x.mean() + scale * (x - x.mean())
    return result, dict(roughness=roughness, jump_q90=jump, original_span=span,
                       target_span=float(target), scale=float(scale),
                       output_span=float(np.ptp(result)))


def main():
    parent = ROOT / "outputs/cpu_local_morph_preview"
    output = parent / "adaptive_v1"
    metadata = json.loads((parent / "manifest.json").read_text())
    source = Path(metadata["source"])
    tracked = [source] + sorted(parent.glob("*"))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.is_file()}
    assert hashes[source] == metadata["source_sha256"]
    assert metadata["config"]["amplitude"] == 1.2
    with np.load(parent / "preview_arrays.npz", allow_pickle=False) as z:
        ids, bases, residuals = (z[k] for k in ("source_rows", "backgrounds", "residuals"))
    np.testing.assert_array_equal(bases, np.load(source, allow_pickle=False)[ids])
    transformed = [compress_background(b) for b in bases]
    backgrounds = np.asarray([b for b, _ in transformed])
    metrics = [m for _, m in transformed]
    samples = backgrounds + residuals
    np.testing.assert_allclose(samples - backgrounds, residuals, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(samples[residuals == 0], backgrounds[residuals == 0])
    for i, (b, m) in enumerate(transformed):
        np.testing.assert_allclose(np.diff(b), m["scale"] * np.diff(bases[i]), atol=1e-14)
        assert 0 < m["scale"] <= 1
    assert np.isfinite(samples).all()
    output.mkdir(parents=True, exist_ok=True)

    def draw(ax, b, i, label):
        case = metadata["cases"][i]
        ax.plot(b, lw=1.3, label="Background")
        ax.plot(b + residuals[i], lw=1.1, label="Background + residual")
        if case["injected_kind"] != "nothing":
            ax.axvspan(case["start"], case["stop_exclusive"] - 1, alpha=.08, color="orange")
        ax.set(title=f"Row {ids[i]} | {case['injected_kind']} | {label}", xlim=(0, 127))
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)

    for group, indices in (("first_4", list(range(4))), ("random_8", list(range(4, 12)))):
        fig, axes = plt.subplots(len(indices)//2, 2, figsize=(15, 3.5*len(indices)//2), squeeze=False)
        for ax, i in zip(axes.flat, indices):
            draw(ax, backgrounds[i], i, f"adaptive scale={metrics[i]['scale']:.3f}")
        fig.suptitle("Adaptive v1 | A=1.2 unchanged | independent y-scales | no filtering")
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(output / f"{group}_overlay.png", dpi=120)
        plt.close(fig)
        fig, axes = plt.subplots(len(indices), 3, figsize=(18, 2.7*len(indices)), squeeze=False, sharey="row")
        for row, i in zip(axes, indices):
            b = bases[i]
            strong = (b-b.min()) / np.ptp(b)*.5-.25 if np.ptp(b) else b.copy()
            for ax, bg, label in zip(row, (b, backgrounds[i], strong), ("Original", "Adaptive v1", "Fixed range 0.5")):
                draw(ax, bg, i, label)
        fig.suptitle("Shared y-limits per row | identical residuals and locations")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output / f"{group}_comparison.png", dpi=120)
        plt.close(fig)
    # Sensitivity comparison holds axes fixed, including nothing cases.
    fig, axes = plt.subplots(12, 3, figsize=(18, 30), squeeze=False, sharey="row")
    for i, row in enumerate(axes):
        for ax, budget in zip(row, (.8, 1., 1.25)):
            b, m = compress_background(bases[i], budget_multiplier=budget)
            draw(ax, b, i, f"budget x{budget}; scale={m['scale']:.3f}")
    fig.suptitle("Sensitivity only: smaller budget = stronger compression | shared y per row")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output / "sensitivity.png", dpi=100)
    plt.close(fig)
    arrays = dict(source_rows=ids, original_backgrounds=bases, backgrounds=backgrounds,
                  residuals=residuals, samples=samples)
    np.savez(output / "preview_arrays.npz", **arrays)
    with np.load(output / "preview_arrays.npz", allow_pickle=False) as z:
        for k, v in arrays.items():
            np.testing.assert_array_equal(z[k], v)
    for p, digest in hashes.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    metadata["background_transform"] = "mean + scale*(background-mean); see rule; no filtering"
    metadata["rule"] = dict(roughness="sum(abs(diff(x)))/ptp(x)",
                            target_span="A*(0.5+2.5*exp(-(max(roughness-4,0)/2.2)^2))",
                            scale="min(1, target_span/ptp(x), 0.2*A/q90(abs(diff(x))))",
                            constant_policy="unchanged", amplitude_reference=1.2)
    metadata["caveats"].append("Heuristic calibrated from 12 visual examples; not a validated general rule. Scalar compression retains relative jaggedness and may flatten trends in mixed signals.")
    for case, m in zip(metadata["cases"], metrics):
        case["compression_metrics"] = m
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    report = dict(status="PASS", cases=len(ids), source_and_previous_previews_unchanged=True,
                  residuals_unchanged=True, saved_array_roundtrip=True,
                  positive_scalar_only=True, amplitude_configuration=1.2)
    (output / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    print([(int(i), round(m["scale"], 3), round(m["output_span"], 3)) for i, m in zip(ids, metrics)], flush=True)


if __name__ == "__main__":
    main()