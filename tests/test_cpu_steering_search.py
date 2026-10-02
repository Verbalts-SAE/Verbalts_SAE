from scripts.search_cpu_steering import candidates, rank


def test_candidates_unique_and_bounded():
    pool = candidates()
    assert len(pool) == len({v['name'] for v in pool}) == 24
    assert all(5 <= v['guidance_t_range'][0] <= v['guidance_t_range'][1] <= 45 for v in pool)


def test_rank_requires_gain_and_quality_on_every_seed():
    pure = dict(accr=.8, mse=1., roughness=1., curvature=1., amplitude_p99=1.)
    configs = [dict(name=n) for n in ('good', 'unsafe', 'flat')]
    report = dict(pure=pure, good=dict(pure, accr=.85),
                  unsafe=dict(pure, accr=.95, amplitude_p99=1.2), flat=pure)
    ranked = rank([report, report], configs)
    assert ranked[0]['config']['name'] == 'good'
    assert ranked[0]['eligible']
    assert not any(r['eligible'] for r in ranked[1:])
    assert rank([report], []) == []