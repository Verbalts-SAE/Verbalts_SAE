import json

from scripts.search_large_steering import REFERENCE, SPLITS, candidates


def test_large_pool_unique_reproducible_and_legal():
    pool = candidates()
    assert len(pool) == 512
    assert pool == candidates()
    signatures = {json.dumps({k: v for k, v in c.items() if k != 'name'}, sort_keys=True)
                  for c in pool}
    assert len(signatures) == 512
    assert all(5 <= c['guidance_t_range'][0] <= c['guidance_t_range'][1] <= 45 for c in pool)
    assert all(c['selection_score'] == 'applied' and c['preserve_residual'] for c in pool)
    assert REFERENCE['max_step'] == 2.


def test_large_splits_and_fresh_tail():
    used = set()
    for offset, count in SPLITS.values():
        indices = set(range(offset, offset + count))
        assert not used.intersection(indices)
        used.update(indices)
    assert max(used) < 2000
    assert SPLITS['fresh'] == (1920, 80)