"""Summarize paired latent round-trip diagnostics without accessing test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sae.provenance import write_json


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty trace")
    keys = ("probability_before", "probability_after", "probability_roundtrip_before",
            "probability_roundtrip_after", "latent_delta_rmse",
            "roundtrip_latent_delta_rmse", "hidden_delta_rmse",
            "hidden_delta_standardized_rmse", "hidden_delta_relative_l2",
            "hidden_reconstruction_rmse")
    result = {k: np.mean([r[k] for r in rows], axis=0).tolist() for k in keys}
    direct = np.asarray(result["probability_after"]) - result["probability_before"]
    retained = np.asarray(result["probability_roundtrip_after"]) - result["probability_roundtrip_before"]
    result.update(records=len(rows), direct_probability_gain=direct.tolist(),
                  roundtrip_probability_gain=retained.tolist())
    # Ratio of aggregate signed gains, not a mean of unstable per-row ratios.
    result["aggregate_gain_retention_ratio"] = [
        float(b / a) if abs(a) > 1e-8 else None for a, b in zip(direct, retained)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    indices = summary["indices"]
    report = {"split": summary["split"], "variants": {},
              "interpretation": "Round-trip loss includes encoder Top-K projection; it is not a decoder-only causal measurement."}
    if args.previous_run:
        previous = json.loads((args.previous_run / "summary.json").read_text())
        for key in ("indices", "seed", "batch_size", "checkpoints", "data_sha256"):
            if summary[key] != previous[key]:
                raise ValueError(f"paired runs differ in {key}")
    baseline = np.load(run / "sae.npy")
    for name, config in summary["variants"].items():
        curves = np.load(run / f"{name}.npy")
        result = {"final_rmse_vs_sae": float(np.sqrt(np.mean((curves - baseline) ** 2)))}
        if args.previous_run:
            old = np.load(args.previous_run / f"{name}.npy")
            result["identical_to_previous_run"] = bool(np.array_equal(curves, old))
            result["max_abs_difference_previous_run"] = float(np.max(np.abs(curves - old)))
        trace = run / f"{name}_trace.json"
        if trace.exists():
            rows = json.loads(trace.read_text())
            lo, hi = config["config"].get("guidance_t_range", [5, 45])
            expected = {(i, t) for i in indices for t in range(lo, hi + 1)}
            observed = {(r["sample_index"], r["timestep"]) for r in rows}
            if observed != expected or len(rows) != len(expected):
                raise ValueError(f"missing or duplicate sample/timestep records: {name}")
            for r in rows:
                if indices[r["sample_position"]] != r["sample_index"]:
                    raise ValueError(f"sample mapping mismatch: {name}")
            result.update(coverage_verified=True, overall=summarize(rows))
            result["by_timestep"] = {str(t): summarize([r for r in rows if r["timestep"] == t])
                                     for t in range(lo, hi + 1)}
            result["by_sample"] = {str(i): summarize([r for r in rows if r["sample_index"] == i])
                                   for i in indices}
        report["variants"][name] = result
    destination = run / "roundtrip_report.json"
    write_json(destination, report)
    print(f"Saved {destination}")
    for name, result in report["variants"].items():
        print(name, json.dumps({k: v for k, v in result.items()
                                if k not in ("by_timestep", "by_sample")}))


if __name__ == "__main__":
    main()