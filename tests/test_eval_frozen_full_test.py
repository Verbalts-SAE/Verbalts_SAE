import json
import pytest

from scripts.eval_frozen_full_test import frozen_configs


def test_requires_completed_search_and_uses_frozen_candidate(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps(dict(state='running')))
    with pytest.raises(ValueError, match='complete'):
        frozen_configs(tmp_path)
    (tmp_path / 'status.json').write_text(json.dumps(dict(state='completed')))
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(extra_baselines=[dict(name='old')])))
    frozen = [dict(name='reference'), dict(name='winner')]
    (tmp_path / 'frozen_final.json').write_text(json.dumps(frozen))
    assert frozen_configs(tmp_path) == [dict(name='pure'), dict(name='sae'), dict(name='old')] + frozen