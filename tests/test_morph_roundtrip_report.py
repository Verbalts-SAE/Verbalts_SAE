import pytest

from sae.report_morph_roundtrip import summarize


def test_paired_roundtrip_control():
    row = dict(probability_before=[.2, .2, .2], probability_after=[.6, .2, .4],
               probability_roundtrip_before=[.1, .1, .1],
               probability_roundtrip_after=[.2, .1, .3])
    for key in ("latent_delta_rmse", "roundtrip_latent_delta_rmse", "hidden_delta_rmse",
                "hidden_delta_standardized_rmse", "hidden_delta_relative_l2",
                "hidden_reconstruction_rmse"):
        row[key] = .01
    result = summarize([row, row])
    assert result["records"] == 2
    assert result["roundtrip_probability_gain"] == pytest.approx([.1, 0, .2])
    assert result["aggregate_gain_retention_ratio"][0] == pytest.approx(.25)
    assert result["aggregate_gain_retention_ratio"][1] is None
    assert result["aggregate_gain_retention_ratio"][2] == pytest.approx(1.)
    with pytest.raises(ValueError, match="empty"):
        summarize([])