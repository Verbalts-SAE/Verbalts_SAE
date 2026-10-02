from scripts.refine_stabilized_steering import REFERENCE, SPLITS, candidates, rank


def test_accr_selection_not_segment_accuracy(monkeypatch):
    from scripts import refine_stabilized_steering as runner
    monkeypatch.setattr(runner, 'PRIMARY_METRIC', 'whole_curve_exact_match')
    def metrics(segment, exact):
        return dict(mse=dict(overall_mse=1.), cnn=dict(segment_accuracy=segment,
                    whole_curve_exact_match=exact), roughness=1., curvature=1.)
    report = dict(variants=dict(reference=metrics(.75, .5),
                               exact_win=metrics(.74, .52), segment_win=metrics(.78, .49)))
    rows = rank([report], [dict(name=n) for n in ['segment_win', 'exact_win']])
    assert rows[0]['config']['name'] == 'exact_win'
    assert rows[0]['eligible'] and not rows[1]['eligible']


def test_unique_grid_and_disjoint_splits():
    pool = candidates()
    assert len(pool) > 40
    assert len({v['name'] for v in pool}) == len(pool)
    used = set(range(768))
    for offset, count in SPLITS.values():
        indices = set(range(offset, offset + count))
        assert not used.intersection(indices)
        used.update(indices)
    assert max(used) < 2000
    assert all(v['preserve_residual'] and v['selection_score'] == 'applied' for v in pool)
    assert all(5 <= v['guidance_t_range'][0] <= v['guidance_t_range'][1] <= 45
               for v in pool)


def test_selection_requires_both_metrics_and_guardrail():
    def metrics(mse, accuracy, curvature=1.):
        return dict(mse=dict(overall_mse=mse), cnn=dict(segment_accuracy=accuracy),
                    roughness=1., curvature=curvature)
    report = dict(variants=dict(reference=metrics(1., .75), good=metrics(.98, .76),
                               rough=metrics(.9, .8, 1.03), bad=metrics(1.1, .9)))
    rows = rank([report], [dict(REFERENCE, name=n) for n in ['rough', 'bad', 'good']])
    assert rows[0]['config']['name'] == 'good'
    assert rows[0]['eligible']
    assert not any(r['eligible'] for r in rows[1:])


def test_small_mse_increase_allowed_accuracy_prioritized():
    def metrics(mse, accuracy):
        return dict(mse=dict(overall_mse=mse), cnn=dict(segment_accuracy=accuracy),
                    roughness=1., curvature=1.)
    report = dict(variants=dict(reference=metrics(1., .75), small=metrics(1.005, .78),
                               lower=metrics(.98, .76), excessive=metrics(1.015, .80),
                               unchanged=metrics(1., .75)))
    rows = rank([report], [dict(REFERENCE, name=n)
                          for n in ['lower', 'small', 'excessive', 'unchanged']])
    assert rows[0]['config']['name'] == 'small'
    assert [r['eligible'] for r in rows] == [True, True, False, False]


def test_per_seed_mse_guardrail():
    reports = []
    for mse in [1.025, .975]:
        reports.append(dict(variants=dict(
            reference=dict(mse=dict(overall_mse=1.), cnn=dict(segment_accuracy=.75),
                           roughness=1., curvature=1.),
            unstable=dict(mse=dict(overall_mse=mse), cnn=dict(segment_accuracy=.78),
                          roughness=1., curvature=1.))))
    assert not rank(reports, [dict(REFERENCE, name='unstable')])[0]['eligible']