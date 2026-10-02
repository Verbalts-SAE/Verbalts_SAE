import numpy as np

from scripts.complete_seed147_report import DATA, RESULTS, SOURCES, digest, read


def test_retained_protocol_and_caption_alignment():
    for seed in (1, 7, 42):
        summary = read(RESULTS / f'priority_test_1011172/seed{seed}/summary.json')
        assert sorted(summary['indices']) == list(range(2000))
        assert summary['seed'] == seed
        assert digest(DATA / 'test_text_caps.npy') == summary['data_sha256']['test_text_caps.npy']
    for family, job, _ in SOURCES.values():
        summary = read(RESULTS / f'steering/{family}_stageD_{job}/seed1/baseline/summary.json')
        assert summary['shape']['samples'] == 2000
        assert digest(DATA / 'test_text_caps.npy') == summary['audit']['captions_sha256']
        config = RESULTS / f'steering_infra/configs/{family}_electricity_v3_clean.yaml'
        assert digest(config) == summary['pairing_fingerprint']['config_sha256']


def test_verbalts_paired_gain_matches_retained_values():
    values = []
    for seed in (1, 7, 42):
        summary = read(RESULTS / f'priority_test_1011172/seed{seed}/summary.json')
        variants = summary['variants']
        values.append([variants[name]['cnn']['whole_curve_exact_match']
                       for name in ('pure', 'previous_reference')])
    values = np.asarray(values)
    delta = 100 * (values[:, 1] - values[:, 0])
    np.testing.assert_allclose(delta.mean(), 1.70)
    np.testing.assert_allclose(delta.std(ddof=1), np.std([1.25, 2.05, 1.8], ddof=1))