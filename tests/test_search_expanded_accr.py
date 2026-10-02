import pytest

from scripts.search_expanded_accr import candidates, signature
from scripts.search_large_steering import candidates as old_candidates


def test_new_pool_unique_reproducible_and_disjoint():
    pool = candidates()
    assert len(pool) == 1024
    assert pool == candidates()
    keys = {signature(c) for c in pool}
    assert len(keys) == 1024
    assert not keys.intersection(signature(c) for c in old_candidates())
    assert max(c['eta'] for c in pool) == 40960
    assert max(c['topk'] for c in pool) == 256
    assert all(5 <= c['guidance_t_range'][0] <= c['guidance_t_range'][1] <= 45 for c in pool)
    assert all(c['preserve_residual'] and c['selection_score'] == 'applied' for c in pool)


@pytest.mark.parametrize('count', [0, 1025])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        candidates(count)