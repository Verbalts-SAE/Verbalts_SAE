"""Re-select completed broad-search candidates on whole-curve exact match, not test."""
import json

from scripts import refine_stabilized_steering as runner
from scripts.search_large_steering import REFERENCE


SOURCE = runner.ROOT / 'results/electricity_v3/large_search_1011120'


def accr_candidates():
    if json.loads((SOURCE / 'status.json').read_text()).get('state') != 'completed':
        raise ValueError('Broad search must be complete')
    rows = []
    for path in sorted(SOURCE.glob('screen_*/summary.json')):
        report = json.loads(path.read_text())
        pool = [v['config'] for k, v in report['variants'].items() if k.startswith('broad_')]
        rows.extend(runner.rank([report], pool))
    return [r['config'] for r in sorted(rows, key=runner.ranking_key) if r['eligible']][:24]


def main():
    runner.PRIMARY_METRIC = 'whole_curve_exact_match'
    runner.EXTRA_BASELINES = [dict(runner.REFERENCE, name='previous_reference')]
    runner.REFERENCE = REFERENCE
    runner.SPLITS = dict(screen=(0, 256), refine=(768, 384), confirm=(1152, 768))
    runner.SHORTLIST = 16
    runner.CAUTION = ('Primary endpoint ACCR = all three segments correct. Reuses broad-search validation; '
                      'not a new independent holdout. No test-based selection. Existing 512-config grid '
                      'includes eta up to 20480, max_step up to 4, rel_cap up to 4. '
                      'Do not reuse the already-consumed fresh 80 cases as a new holdout.')
    runner.candidates = accr_candidates
    runner.main()


if __name__ == '__main__':
    main()