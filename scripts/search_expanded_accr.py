"""Expanded ACCR-only validation search; observed test is never used to select."""
import json
import random

from scripts import refine_stabilized_steering as runner
from scripts.search_large_steering import REFERENCE, candidates as old_candidates


def signature(config):
    return json.dumps({k: v for k, v in config.items() if k != 'name'}, sort_keys=True)


def candidates(count=1024):
    if not 1 <= count <= 1024:
        raise ValueError('count must be between 1 and 1024')
    rng = random.Random(20260925)
    choices = dict(
        topk=[4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        eta=[640, 1280, 2560, 5120, 8192, 10240, 20480, 40960],
        gamma=[0, .25, .5, 1., 2., 3.],
        rel_cap=[.5, 1., 1.5, 2., 2.5, 3., 4., 6.],
        max_step=[.25, .5, 1., 1.5, 2., 3., 4., 6.],
        guidance_t_range=[[5, 45], [5, 35], [5, 25], [10, 45], [10, 35],
                          [15, 45], [20, 45], [25, 45], [30, 45], [5, 15], [15, 35]])
    seen = {signature(c) for c in old_candidates()}
    seen.update([signature(REFERENCE), signature(dict(REFERENCE, max_step=1.))])
    pool = []
    while len(pool) < count:
        config = dict(REFERENCE)
        keys = rng.sample(list(choices), 3) if len(pool) < 512 else list(choices)
        for key in keys:
            config[key] = rng.choice(choices[key])
        key = signature(config)
        if key in seen:
            continue
        seen.add(key)
        config['name'] = f'expanded_{len(pool):04d}'
        pool.append(config)
    return pool


def configure():
    runner.PRIMARY_METRIC = 'whole_curve_exact_match'
    runner.EXTRA_BASELINES = [dict(REFERENCE, name='previous_reference', max_step=1.)]
    runner.REFERENCE = REFERENCE
    runner.SPLITS = dict(screen=(0, 256), refine=(768, 384), confirm=(1152, 768))
    runner.REFINE_SEEDS = (1, 7, 42)
    runner.CONFIRM_SEEDS = (1, 7, 42)
    runner.SHORTLIST = 24
    runner.CAUTION = ('ACCR means all three stages correct. All validation partitions are reused; '
                      'confirmation is an audit, not an independent holdout. Test has already been '
                      'observed in prior experiments and is excluded from this search. '
                      'Seeds share samples. No automatic test evaluation or test-based selection.')
    runner.candidates = candidates


if __name__ == '__main__':
    configure()
    runner.main()