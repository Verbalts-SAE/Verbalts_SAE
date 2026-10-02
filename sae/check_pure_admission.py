"""Enforce pure-model admission criteria before any SAE steering run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(report: dict, variance_bounds=(.8, 1.25), extreme_ratio_max=1.5) -> dict:
    correct = report["correct"]["cnn"]
    sensitivity = report["condition_sensitivity"]
    swaps = report["fixed_noise_caption_swap"]["groups"]
    swap_total = sum(len(row["target_middle"]) for row in swaps)
    swap_correct = sum(sum(a == b for a, b in zip(row["target_middle"], row["middle_predictions"]))
                       for row in swaps)
    truth, generated = report["distribution"]["truth"], report["distribution"]["generated"]
    variance_ratio = generated["variance"] / max(truth["variance"], 1e-12)
    extreme_ratio = (generated["absolute_extreme_quantiles"]["0.99"] /
                     max(truth["absolute_extreme_quantiles"]["0.99"], 1e-12))
    checks = {
        "shuffle_scored_against_original_condition": (
            sensitivity.get("scoring_target") == "original_condition"
        ),
        "segment_accuracy": correct["segment_accuracy"] >= .85,
        "whole_curve_exact_match": correct["whole_curve_exact_match"] >= .60,
        "correct_significantly_better_than_shuffled": sensitivity["bootstrap_ci95"][0] > 0,
        "fixed_noise_swap_middle_accuracy": swap_total > 0 and swap_correct / swap_total >= .85,
        "variance_ratio": variance_bounds[0] <= variance_ratio <= variance_bounds[1],
        "absolute_extreme_p99_ratio": extreme_ratio <= extreme_ratio_max,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"segment_accuracy": correct["segment_accuracy"],
                       "whole_curve_exact_match": correct["whole_curve_exact_match"],
                       "condition_accuracy_delta_ci95": sensitivity["bootstrap_ci95"],
                       "swap_middle_accuracy": swap_correct / max(swap_total, 1),
                       "variance_ratio": variance_ratio, "absolute_extreme_p99_ratio": extreme_ratio},
            "thresholds": {"segment_accuracy": .85, "whole_curve_exact_match": .60,
                           "variance_ratio": list(variance_bounds),
                           "absolute_extreme_p99_ratio_max": extreme_ratio_max}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.summary.read_text()))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()