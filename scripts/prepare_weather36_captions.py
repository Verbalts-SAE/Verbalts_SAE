"""Export weather36 training data with train-only, background-based captions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("slope", "standard_deviation", "mean_abs_change", "cid_ce_normalized", "autocorrelation_lag_1")
SPLITS = ("train", "valid", "test")


def features(backgrounds):
    """Constant backgrounds have undefined autocorrelation: retain NaN for audit."""
    return np.asarray([[fc.linear_trend(x, [{"attr": "slope"}])[0][1],
                        fc.standard_deviation(x), fc.mean_abs_change(x),
                        fc.cid_ce(x, normalize=True), fc.autocorrelation(x, 1)]
                       for x in backgrounds], dtype=float)


def fit_thresholds(train):
    if not np.isfinite(train[:, :3]).all():
        raise ValueError("nonfinite caption features")
    return {"flat_slope": max(1e-8, float(np.quantile(np.abs(train[:, 0]), .2))),
            "steep_slope": max(1e-8, float(np.quantile(np.abs(train[:, 0]), .75))),
            "variability": np.quantile(train[:, 1], [1/3, 2/3]).tolist(),
            "changes": np.quantile(train[:, 2], [1/3, 2/3]).tolist()}


def background_caption(row, thresholds):
    slope, variability, changes = row[:3]
    if abs(slope) <= thresholds["flat_slope"]:
        trend = "is mostly flat"
    else:
        trend = ("rises" if slope > 0 else "falls") + (
            " gradually" if abs(slope) <= thresholds["steep_slope"] else " noticeably")
    level = lambda value, key: int(np.searchsorted(thresholds[key], value, side="left"))
    variability = ("low", "moderate", "fairly high")[level(variability, "variability")]
    changes = ("smooth", "moderate", "fairly brisk")[level(changes, "changes")]
    return f"The background {trend} overall, with {variability} variability and {changes} point-to-point changes."


def local_caption(labels):
    phrases = {"single_peak": "a single peak", "double_peaks": "double peaks", "sag": "a sag"}
    return " ".join(f"The {stage} part has {phrases[label]}."
                    for stage, label in zip(("beginning", "middle", "end"), labels)
                    if label != "nothing")


def prepare(source, output):
    metadata = json.loads((source / "metadata.json").read_text())
    if not (source / "validation_report.txt").read_text().startswith("PASS:"):
        raise ValueError("source audit did not pass")
    if metadata["sequence_length"] != 36 or metadata["combined_total"] != 40500:
        raise ValueError("unexpected source dataset")
    matrices = {split: features(np.load(source / split / "backgrounds.npy")) for split in SPLITS}
    thresholds = fit_thresholds(matrices["train"])
    train = matrices["train"]
    finite = np.isfinite(train).all(axis=1)
    report = {"features": FEATURES, "threshold_fit_split": "train",
              "independent_training_backgrounds": len(train), "thresholds": thresholds,
              "selected": list(FEATURES[:3]),
              "selection_reason": "Compact trend/amplitude/smoothness baseline; CID and lag-1 retained for audit, not verbalized to avoid redundant short-sequence descriptors.",
              "nonfinite_counts": {s: (~np.isfinite(m)).sum(axis=0).tolist() for s, m in matrices.items()},
              "train_quantiles": {name: np.nanquantile(train[:, i], [0, .1, .5, .9, 1]).tolist() for i, name in enumerate(FEATURES)},
              "train_correlation_complete_cases": np.corrcoef(train[finite].T).tolist()}
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        records = json.loads((source / split / "combined_records.json").read_text())
        samples = np.load(source / split / "combined_samples.npy")
        labels = np.load(source / split / "combined_labels.npy")
        ids = np.asarray([r["background_id"] for r in records])
        assert samples.shape == (len(records), 36, 1) and labels.shape == (len(records), 3)
        local = [local_caption(r["morphologies"]) for r in records]
        global_caps = [background_caption(row, thresholds) for row in matrices[split]]
        caps = [f"{text} {global_caps[bg]}" for text, bg in zip(local, ids)]
        for suffix, array in {"ts": samples.astype(np.float32), "attrs_idx": labels,
                              "text_caps": np.asarray(caps)[:, None],
                              "text_caps_local": np.asarray(local)[:, None],
                              "background_ids": ids, "background_features": matrices[split]}.items():
            np.save(output / f"{split}_{suffix}.npy", array)
        report.setdefault("examples", {})[split] = [caps[i] for i in (0, 13, 26, len(caps)//2)]
    metadata.update(source_directory=str(source), caption_style="morphology-first-background-three-feature",
                    caption_thresholds=thresholds, stage_names=["beginning", "middle", "end"],
                    n_var=1, seq_length=36, normalize=False)
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "feature_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def embed(output, device):
    from contsg.data.precompute.sentence_transformer import SentenceTransformerPrecomputer
    model_path = "/public/home/liym2024/models/Qwen3-Embedding-0.6B"
    encoder = SentenceTransformerPrecomputer(model_path=model_path, embed_dim=1024, device=device)
    provenance = {}
    for split in SPLITS:
        for variant in ("", "_local"):
            path = output / f"{split}_text_caps{variant}.npy"
            captions = np.load(path).reshape(-1)
            unique, inverse = np.unique(captions, return_inverse=True)
            values = encoder.compute(unique.tolist(), batch_size=16)[inverse]
            assert values.shape == (len(captions), 1024) and np.isfinite(values).all()
            target = output / f"{split}_cap_emb{variant}.npy"
            np.save(target, values.astype(np.float32))
            provenance[f"{split}{variant}"] = {"caption_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "embedding_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "encoder": model_path, "shape": list(values.shape), "unique_captions": len(unique)}
    (output / "embedding_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/weather_three_stage_36_morph_40k")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/weather_three_stage_36_morph")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.embed_only:
        embed(args.output, args.device)
    else:
        print(json.dumps(prepare(args.source, args.output), indent=2))