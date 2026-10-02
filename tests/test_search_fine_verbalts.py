import json

from scripts.search_fine_verbalts import (
    REFERENCE, committed, digest, local_grid, shortlist, signature, windows,
)


def test_integer_windows():
    pool = windows()
    assert len(pool) == len({signature(c) for c in pool}) == 861
    assert sum(c['guidance_t_range'][0] == c['guidance_t_range'][1] for c in pool) == 41
    assert all(c['preserve_residual'] and c['gamma'] == 0 for c in pool)
    assert sum(signature(c) == signature(REFERENCE) for c in pool) == 1


def test_local_budget_and_deduplication():
    centres = windows()[:8]
    strength = local_grid(centres, 'strength')
    caps = local_grid(strength[:12], 'caps')
    assert len(strength) == 200
    assert len(caps) <= 300
    assert len(local_grid([REFERENCE, REFERENCE], 'strength')) == 25
    assert len({signature(c) for c in caps}) == len(caps)


def test_diverse_shortlist():
    rows = [dict(config=c) for c in windows()]
    selected = shortlist(rows)
    assert len(selected) == 40
    assert len({signature(c) for c in selected}) == 40
    assert selected[:20] == windows()[:20]


def test_resume_integrity(tmp_path):
    report = tmp_path / 'summary.json'
    marker = tmp_path / 'complete.json'
    report.write_text('{}')
    assert not committed(report, marker)
    marker.write_text(json.dumps({'summary.json': digest(report)}))
    assert committed(report, marker)
    report.write_text('{"changed": true}')
    assert not committed(report, marker)