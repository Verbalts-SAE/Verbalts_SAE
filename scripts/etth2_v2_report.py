"""Cross-model v2 steering report for etth2_2class.

Scans the latest v2_fulltest steering summaries for
seed{1,7,42} x {verbalts_v2,diffusets,bridge}, aggregates ACCR / MSE per
variant across seeds, and compares against the v1 verbalts baseline
(latest steering run under seed*/verbalts/steering).

Usage: python scripts/etth2_v2_report.py [--print-table]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/etth2_2class"
MODELS = ("verbalts_v2", "diffusets", "bridge")
SEEDS = (1, 7, 42)


def load_latest(root: Path, pattern: str) -> dict | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text())


def accr_of(variant: dict) -> float:
    cnn = variant.get("cnn", {})
    return float(cnn.get("whole_curve_exact_match", float("nan")))


def mse_of(variant: dict) -> float:
    mse = variant.get("mse", {})
    return float(mse.get("overall_mse", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()
    report: dict = {"per_model": {}, "v1_baseline": {}}
    rows = []
    for model in MODELS:
        model_report: dict = {"seeds": {}, "cross_seed": {}}
        for seed in SEEDS:
            steering_dir = RES / f"seed{seed}" / model / "steering"
            summary = load_latest(steering_dir, "v2_fulltest_*/summary.json")
            if summary is None:
                continue
            variants = summary["variants"]
            pure = accr_of(variants["pure"])
            best_name, best_delta = None, float("-inf")
            for name, v in variants.items():
                if name in ("pure", "sae"):
                    continue
                delta = accr_of(v) - pure
                if delta > best_delta:
                    best_delta, best_name = delta, name
            seed_entry = {
                "pure_accr": pure,
                "pure_mse": mse_of(variants["pure"]),
                "best_variant": best_name,
                "best_accr": accr_of(variants[best_name]) if best_name else None,
                "best_delta": best_delta if best_name else None,
                "best_mse": mse_of(variants[best_name]) if best_name else None,
                "sae_accr": accr_of(variants.get("sae", {})),
                "variants": {n: {"accr": accr_of(v), "mse": mse_of(v)}
                             for n, v in variants.items()},
            }
            model_report["seeds"][str(seed)] = seed_entry
            rows.append((model, seed, seed_entry))
        if model_report["seeds"]:
            agg = {}
            for key in ("pure_accr", "pure_mse", "best_accr", "best_delta", "best_mse", "sae_accr"):
                vals = [s[key] for s in model_report["seeds"].values()
                        if s.get(key) is not None]
                agg[key] = sum(vals) / len(vals) if vals else None
            model_report["cross_seed"] = agg
        report["per_model"][model] = model_report
    # v1 verbalts baseline (latest steering run under seed*/verbalts/steering)
    for seed in SEEDS:
        steering_dir = RES / f"seed{seed}" / "verbalts" / "steering"
        summary = load_latest(steering_dir, "*/summary.json")
        if summary is None:
            continue
        variants = summary["variants"]
        pure = accr_of(variants["pure"])
        best_delta, best_name = float("-inf"), None
        for name, v in variants.items():
            if name in ("pure", "sae"):
                continue
            delta = accr_of(v) - pure
            if delta > best_delta:
                best_delta, best_name = delta, name
        report["v1_baseline"][str(seed)] = {
            "pure_accr": pure,
            "best_variant": best_name,
            "best_delta": best_delta if best_name else None,
        }
    out = RES / "steering/v2_cross_model_report.json"
    out.write_text(json.dumps(report, indent=2))
    if args.print_table:
        print(f"{'model':<14}{'seed':<6}{'pure':<8}{'best':<8}{'dACC':<8}"
              f"{'best_var':<20}{'pure_mse':<10}{'best_mse':<10}")
        for model, seed, entry in rows:
            print(f"{model:<14}{seed:<6}{entry['pure_accr']:<8.4f}"
                  f"{entry['best_accr']:<8.4f}{entry['best_delta']:<8.4f}"
                  f"{str(entry['best_variant']):<20}"
                  f"{entry['pure_mse']:<10.4f}{entry['best_mse']:<10.4f}")
        for model in MODELS:
            agg = report["per_model"].get(model, {}).get("cross_seed", {})
            if agg:
                print(f"{model:<14}{'avg':<6}{agg['pure_accr']:<8.4f}"
                      f"{agg['best_accr']:<8.4f}{agg['best_delta']:<8.4f}")
        for seed, v in report["v1_baseline"].items():
            print(f"{'v1 verbalts':<14}{seed:<6}{v['pure_accr']:<8.4f}"
                  f"{'':<8}{v['best_delta']:<8.4f}{str(v['best_variant']):<20}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
