"""Preview gentler background compression without modifying existing datasets."""
from copy import deepcopy
import json

import numpy as np

from scripts.generate_weather_morph_preview import ROOT, BOUNDS, plt


def rescale_background(background, record):
    result = np.empty_like(background, dtype=np.float64)
    updated = deepcopy(record)
    for segment in updated["segments"]:
        start, stop = segment["output_start"], segment["output_stop"]
        old_scale = segment["scale"]
        if not np.isfinite(old_scale) or old_scale <= 0:
            raise ValueError("Source scale must be finite and positive")
        gentle = segment["max_step_ratio"] <= 0.25 and segment["total_variation_ratio"] <= 2.5
        scale = 1.0 if segment["channel"] == "air_pressure" else (0.4 if gentle else 0.2)
        anchor = result[start - 1] if start else 0.0
        result[start:stop] = (background[start:stop] - background[start]) * (scale / old_scale) + anchor
        segment.update(scale=scale, anchor=float(anchor))
    return result, updated


def main():
    source = ROOT / "outputs/weather_four_stage_morph_40k_soft_noise"
    out = ROOT / "outputs/weather_relaxed_scale_preview"
    cases = json.loads((source / "different_combinations_cases.json").read_text())
    records = json.loads((source / "test/combined_records.json").read_text())
    background_records = json.loads((source / "test/background_records.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["amplitude"] == 1.2 and metadata["observation_noise"]["std"] == 0.06
    backgrounds = np.load(source / "test/backgrounds.npy")
    old = np.load(source / "test/combined_samples.npy", mmap_mode="r")
    morphology = np.load(source / "test/combined_morphology_residuals.npy", mmap_mode="r")
    noise = np.load(source / "test/combined_noise.npy", mmap_mode="r")
    indices = {r["sample_id"]: i for i, r in enumerate(records)}
    curves, bases, noises, provenance = [], [], [], []
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        i, bid = indices[case["sample_id"]], case["background_id"]
        base, record = rescale_background(backgrounds[bid], background_records[bid])
        new_noise = noise[i, :, 0] * (0.08 / 0.06)
        curve = base + morphology[i, :, 0] + new_noise
        assert np.isfinite(curve).all()
        for boundary in (36, 72, 108):
            np.testing.assert_equal(base[boundary], base[boundary - 1])
        curves.append(curve)
        bases.append(base)
        noises.append(new_noise)
        provenance.append(record)
        ax.plot(backgrounds[bid], "--", color="0.7", label="Previous background (0.3 / 0.1)")
        ax.plot(base, "--", color="0.3", label="New background (0.4 / 0.2)")
        ax.plot(old[i, :, 0], color="tab:orange", alpha=0.65, label="Previous sample: noise 0.06")
        ax.plot(curve, color="tab:blue", linewidth=1.2, label="New sample: noise 0.08")
        for label, (start, stop) in zip(case["morphologies"], BOUNDS):
            if start:
                ax.axvline(start, color="0.75", linewidth=0.8)
            ax.text((start + stop - 1) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set(title=f"Sample {case['sample_id']} | background {bid}", xlim=(0, 127), xlabel="Sample index")
        ax.margins(y=0.25)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="lower left")
    out.mkdir(exist_ok=False)
    fig.suptitle("Less compressed weather backgrounds; morphology amplitude 1.2; noise std 0.08")
    fig.savefig(out / "different_combinations.png", dpi=140)
    plt.close(fig)
    np.save(out / "samples.npy", np.asarray(curves)[..., None])
    np.save(out / "backgrounds.npy", bases)
    np.save(out / "noise.npy", np.asarray(noises)[..., None])
    (out / "metadata.json").write_text(json.dumps({
        "purpose": "Six-case preview only, not a complete training dataset",
        "parent_dataset": str(source), "amplitude": 1.2,
        "noise_std": 0.08, "noise_policy": "Parent saved noise multiplied by 0.08/0.06",
        "parent_observation_noise": metadata["observation_noise"],
        "cases": cases, "background_records": provenance,
    }, indent=2) + "\n")
    print(f"PASS: six finite samples, continuous backgrounds; saved to {out}")


if __name__ == "__main__":
    main()