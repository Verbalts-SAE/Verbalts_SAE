from scripts import eval_quick_frozen_test as runner
import json


def test_quick_configs_freezes_named_candidate(tmp_path, monkeypatch):
    source = tmp_path / 'results/electricity_v3/large_search_1011120'
    source.mkdir(parents=True)
    configs = [dict(name=n, eta=5120) for n in
               ['pure', 'sae', 'reference', 'previous_reference', 'broad_060', 'broad_061']]
    (source / 'screen_060.json').write_text(json.dumps(configs))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    actual = runner.quick_configs()
    assert [c['name'] for c in actual] == ['pure', 'sae', 'reference', 'previous_reference', 'broad_060']
    assert actual[-1]['eta'] == 5120