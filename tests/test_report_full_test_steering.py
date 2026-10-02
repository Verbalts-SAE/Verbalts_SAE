import numpy as np

from scripts.report_full_test_steering import mean_std, select_cases, draw_case, correct_matrix


def test_sample_std_and_case_groups():
    assert mean_std([1, 2, 3]) == dict(mean=2., std=1.)
    assert mean_std([1])['std'] is None
    p = np.array([[True]*3, [False]*3, [True]*3, [False]*3])
    s = np.array([[False]*3, [True]*3, [True]*3, [False]*3])
    groups = select_cases(p, s)
    assert groups['fixed_by_steering'] == [1]
    assert groups['broken_by_steering'] == [0]
    assert groups['both_incorrect'] == [3]
    assert groups['both_correct'] == [2]


def test_plot_smoke_and_actual_predictions(tmp_path):
    pred = [dict(shape='nothing', joint_confidence=.8) for _ in range(3)]
    targets = ['nothing', 'single peak', 'sag']
    assert correct_matrix([pred], [targets]).tolist() == [[True, False, False]]
    path = tmp_path / 'case.png'
    draw_case(path, {k: np.sin(np.arange(128) / 10) for k in ['GT', 'VerbalTS', 'Steering']},
              {k: pred for k in ['GT', 'VerbalTS', 'Steering']}, targets, 'Synthetic smoke test')
    assert path.stat().st_size > 1000