"""Build the approved CPU pilot recipe without fitting new caption thresholds."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.preview_cpu_adaptive_compression import compress_background
from scripts.preview_cpu_background_denoising import denoise_background
from scripts.preview_cpu_local_morph import INTERVALS, KINDS, local_residual
from scripts.preview_cpu_shape_captions import RULE, trajectory, local_caption, validate_caption
from contsg.data.datasets.semisynth_morph import InjectionConfig, MORPHOLOGY_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets/cpu_shape_v1"


def build(source, output, seed=42):
    source, output = Path(source), Path(output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = np.load(source, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 128 or not np.isfinite(data).all():
        raise ValueError("Expected finite length-128 backgrounds")
    # Fail rather than silently allow identical backgrounds into different splits.
    if len(np.unique(data, axis=0)) != len(data):
        raise ValueError("Duplicate backgrounds require group-aware splitting")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    boundaries = [int(.8 * len(data)), int(.9 * len(data))]
    splits = dict(zip(("train", "valid", "test"), np.split(indices, boundaries)))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(source), source_sha256=digest, seed=seed,
                  rule=RULE, amplitude=1.2, detail_retention=.5, splits={},
                  caveats=["Frozen pilot heuristics; not training-calibrated.",
                           "Source entity/time metadata unavailable; row split only.",
                           "Local labels describe injections, not certified final morphology."])
    for split, ids in splits.items():
        # Four kinds equally represented; positions balanced within each kind.
        assignments = rng.permutation(np.arange(len(ids)) % 12)
        backgrounds, residuals, captions, records, attrs = [], [], [], [], []
        for row, assignment in zip(ids, assignments):
            kind, stage = KINDS[assignment % 4], int(assignment // 4)
            start, stop = INTERVALS[stage]
            bg = denoise_background(compress_background(data[row])[0], .5)
            residual = local_residual(kind, start, stop, InjectionConfig(amplitude=1.2))
            case = dict(source_row=int(row), injected_kind=kind, start=start,
                        stop_exclusive=stop)
            diagnostic = trajectory(bg)
            caption = (local_caption(case, 128) + " " + diagnostic["caption"]).strip()
            validate_caption(caption)
            label = [0, 0, 0]
            label[stage] = MORPHOLOGY_NAMES.index(kind)
            backgrounds.append(bg)
            residuals.append(residual)
            captions.append([caption])
            attrs.append(label)
            records.append(dict(**case, caption=caption, trajectory=diagnostic))
        bg, residual = np.asarray(backgrounds), np.asarray(residuals)
        ts = (bg + residual).astype(np.float32)[..., None]
        assert np.isfinite(ts).all()
        arrays = dict(ts=ts, backgrounds=bg, residuals=residual,
                      text_caps=np.asarray(captions), attrs_idx=np.asarray(attrs, dtype=np.int64),
                      source_indices=ids)
        for key, value in arrays.items():
            path = output / f"{split}_{key}.npy"
            np.save(path, value)
            np.testing.assert_array_equal(np.load(path), value)
        (output / f"{split}_records.json").write_text(json.dumps(records, indent=2))
        report["splits"][split] = dict(size=len(ids),
            kinds=dict(Counter(r["injected_kind"] for r in records)),
            trajectories=dict(Counter(r["trajectory"]["shape"] for r in records)))
        print(split, report["splits"][split], flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report.update(status="PASS", source_unchanged=True, disjoint_source_rows=True,
                  no_exact_duplicate_backgrounds=True)
    report["hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in output.glob("*.npy")}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    (output / "meta.json").write_text(json.dumps(dict(seq_length=128, n_var=1,
        name="cpu_shape_v1", split_sizes={k: len(v) for k, v in splits.items()}), indent=2))
    return report


if __name__ == "__main__":
    build(ROOT / "datasets/CPU/daily128_z.npy", OUTPUT)