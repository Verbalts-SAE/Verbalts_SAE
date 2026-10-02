"""Aggregate etth2_2class steering results across seeds into a final report.

Reads:
- results/etth2_2class/cnn_test_report.json (CNN audit + retrained test ACCR)
- results/etth2_2class/seed{1,7,42}/steering/fulltest_*/summary.json (standard)
- results/etth2_2class/seed{1,7,42}/steering/amplified_fulltest_*/summary.json (amplified)

Prints a per-seed and cross-seed summary: ACCR (whole-curve exact match),
MSE, and paired deltas vs pure on the FULL test split (3533 samples).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RES = Path("/public/home/liym2024/Verbalts_SAE/results/etth2_2class")


def load_variants(summary: Path) -> dict:
    data = json.loads(summary.read_text())
    out = {}
    for name, metrics in data["variants"].items():
        out[name] = {
            "accr": metrics["cnn"]["whole_curve_exact_match"],
            "seg_acc": metrics["cnn"]["segment_accuracy"],
            "mse": metrics["mse"]["overall_mse"],
            "d_mse": metrics["mse"]["paired_delta_vs_pure_mean"],
            "roughness": metrics["roughness"],
            "curvature": metrics["curvature"],
        }
    return out


def best_variant(variants: dict, max_mse_rel_worsening: float = 0.01) -> str | None:
    """Best ACCR among variants whose MSE does not worsen by more than 1% vs pure."""
    pure_mse = variants["pure"]["mse"]
    candidates = [
        name
        for name, v in variants.items()
        if name != "pure"
        and v["mse"] <= pure_mse * (1.0 + max_mse_rel_worsening)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda name: variants[name]["accr"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-mse-rel-worsening", type=float, default=0.01,
                        help="relative MSE worsening tolerance vs pure (default 1%%)")
    args = parser.parse_args()

    cnn_report = json.loads((RES / "cnn_test_report.json").read_text())
    print("=" * 90)
    print("CNN (segment classifier) on FULL test (3533 samples)")
    print("=" * 90)
    audit = cnn_report["external_checkpoint_audit"]
    print(f"external etth2_2class_best.pt: seg_acc={audit['segment_accuracy']:.4f} "
          f"ACCR={audit['whole_curve_exact_match_accr']:.4f} "
          f"(claimed 0.972 -> verified {audit['segment_accuracy']:.4f})")
    for seed, report in cnn_report["retrained_cnns"].items():
        print(f"retrained {seed}: seg_acc={report['segment_joint_accuracy']:.4f} "
              f"ACCR={report['whole_curve_exact_match_accr']:.4f}")

    print()
    print("=" * 90)
    print("Steering on FULL test (paired pure/sae/guided, n=3533)")
    print("=" * 90)
    all_seeds = {}
    for seed in ("1", "7", "42"):
        summaries = sorted(glob.glob(str(RES / f"seed{seed}" / "steering" / "*fulltest_*" / "summary.json")))
        if not summaries:
            print(f"seed{seed}: no steering summaries found")
            continue
        merged = {}
        for summary in summaries:
            merged.update(load_variants(Path(summary)))
        pure = merged["pure"]
        print(f"\n--- seed {seed} (pure ACCR={pure['accr']:.4f}, MSE={pure['mse']:.4f}) ---")
        for name in sorted(merged):
            v = merged[name]
            if name == "pure":
                continue
            print(f"  {name:22s} ACCR={v['accr']:.4f} ({v['accr'] - pure['accr']:+.4f}) "
                  f"seg_acc={v['seg_acc']:.4f} MSE={v['mse']:.4f} ({v['d_mse']:+.4f})")
        best = best_variant(merged, args.max_mse_rel_worsening)
        if best is not None:
            b = merged[best]
            print(f"  >>> best within {args.max_mse_rel_worsening:.0%} MSE tolerance: "
                  f"{best} ACCR={b['accr']:.4f} (+{b['accr'] - pure['accr']:.4f}) "
                  f"MSE={b['mse']:.4f} ({b['mse'] / pure['mse'] - 1:+.2%})")
        all_seeds[seed] = merged

    if len(all_seeds) == 3:
        print("\n" + "=" * 90)
        print("Cross-seed summary (mean over seeds 1/7/42, FULL test)")
        print("=" * 90)
        names = sorted(set().union(*[set(v) for v in all_seeds.values()]))
        print(f"{'variant':22s} {'ACCR':>8s} {'dACCR':>8s} {'MSE':>8s} {'dMSE':>8s}")
        pure_accrs = [all_seeds[seed]['pure']['accr'] for seed in all_seeds]
        pure_mses = [all_seeds[seed]['pure']['mse'] for seed in all_seeds]
        for name in names:
            if name == 'pure':
                continue
            accrs, mses = [], []
            for seed, merged in all_seeds.items():
                if name not in merged:
                    continue
                accrs.append(merged[name]['accr'])
                mses.append(merged[name]['mse'])
            mean_daccr = float(np.mean([a - p for a, p in zip(accrs, pure_accrs)]))
            mean_dmse = float(np.mean([m - p for m, p in zip(mses, pure_mses)]))
            print(f"{name:22s} {np.mean(accrs):8.4f} {mean_daccr:+8.4f} "
                  f"{np.mean(mses):8.4f} {mean_dmse:+8.4f}")
        print(f"pure mean ACCR: {np.mean(pure_accrs):.4f}, MSE: {np.mean(pure_mses):.4f}")


if __name__ == "__main__":
    main()
