from sae.check_pure_admission import evaluate


def _report(segment=.9, exact=.7, ci=(.02, .1), swap_correct=True,
            truth_variance=1., generated_variance=1., extreme=1.):
    targets = ["nothing", "single peak", "double peaks", "sag"]
    return {
        "correct": {"cnn": {"segment_accuracy": segment,
                              "whole_curve_exact_match": exact}},
        "condition_sensitivity": {
            "bootstrap_ci95": list(ci),
            "scoring_target": "original_condition",
        },
        "fixed_noise_caption_swap": {"groups": [{
            "target_middle": targets,
            "middle_predictions": targets if swap_correct else ["nothing"] * 4,
        }]},
        "distribution": {
            "truth": {"variance": truth_variance,
                      "absolute_extreme_quantiles": {"0.99": 2.}},
            "generated": {"variance": generated_variance,
                          "absolute_extreme_quantiles": {"0.99": 2. * extreme}},
        },
    }


def test_admission_passes_only_when_every_gate_passes():
    result = evaluate(_report())
    assert result["passed"]
    assert all(result["checks"].values())


def test_admission_blocks_weak_condition_sensitivity_and_amplitude():
    result = evaluate(_report(ci=(-.01, .1), generated_variance=1.5, extreme=2.))
    assert not result["passed"]
    assert not result["checks"]["correct_significantly_better_than_shuffled"]
    assert not result["checks"]["variance_ratio"]
    assert not result["checks"]["absolute_extreme_p99_ratio"]


def test_admission_rejects_legacy_shuffle_scoring():
    report = _report()
    report["condition_sensitivity"].pop("scoring_target")
    result = evaluate(report)
    assert not result["passed"]
    assert not result["checks"]["shuffle_scored_against_original_condition"]