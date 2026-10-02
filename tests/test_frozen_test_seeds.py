import pytest

from scripts.eval_frozen_full_test import validate_seeds


def test_requested_test_seeds():
    assert validate_seeds([1, 11, 42]) == [1, 11, 42]


@pytest.mark.parametrize('seeds', [[1, 1, 42], [-1, 11, 42]])
def test_reject_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        validate_seeds(seeds)