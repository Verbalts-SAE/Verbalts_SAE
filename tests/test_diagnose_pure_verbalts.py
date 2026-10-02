import numpy as np

from sae.diagnose_pure_verbalts import (
    caption_swap_groups,
    condition_comparison,
    curve_statistics,
    mse_summary,
)


def _caption(beginning: str, middle: str, end: str, global_text: str = "Global low") -> str:
    return (f"{global_text}. Beginning morphology: {beginning}. "
            f"Middle morphology: {middle}. End morphology: {end}.")


def test_caption_swap_groups_require_identical_non_middle_context():
    shapes = ("nothing", "single peak", "double peaks", "sag")
    captions = np.array([_caption("sag", shape, "nothing") for shape in shapes] +
                        [_caption("sag", shape, "nothing", "Global high") for shape in shapes[:3]])
    groups = caption_swap_groups(captions)
    assert len(groups) == 1
    assert [groups[0][shape] for shape in shapes] == [0, 1, 2, 3]


def test_diagnostic_summaries_use_per_curve_values():
    truth = np.zeros((2, 3))
    curves = np.array([[1., 1., 1.], [0., 2., 4.]])
    np.testing.assert_allclose(mse_summary(curves, truth)["mean"], 23.0 / 6.0)
    stats = curve_statistics(curves)
    assert stats["range_quantiles"]["1"] == 4.0
    assert stats["absolute_extreme_quantiles"]["1"] == 4.0


def _predictions(names):
    return [[{"shape": shape} for shape in row] for row in names]


def test_condition_comparison_scores_shuffle_against_original_targets():
    original = np.array([
        ["nothing", "single peak", "double peaks"],
        ["sag", "double peaks", "single peak"],
        ["single peak", "nothing", "sag"],
        ["double peaks", "sag", "nothing"],
    ])
    shuffled_input = original[[1, 0, 3, 2]]
    report = condition_comparison(
        original,
        shuffled_input,
        _predictions(original),
        _predictions(shuffled_input),
        seed=7,
        
    )
    assert report["scoring_target"] == "original_condition"
    assert report["shuffled_input_adherence_segment_accuracy"] == 1.0
    assert report["shuffled_input_adherence_exact_match"] == 1.0
    assert report["shuffled_original_target_segment_accuracy"] < 1.0
    assert report["mean"] > 0
    assert report["whole_curve_exact_match"]["mean"] == 1.0