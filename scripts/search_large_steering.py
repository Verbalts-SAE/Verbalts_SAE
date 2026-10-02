"""Deterministic broad search; reused validation audit is not a fresh holdout."""
import random

from scripts import refine_stabilized_steering as runner


REFERENCE = dict(runner.REFERENCE, max_step=2.)
SPLITS = dict(screen=(0, 256), refine=(768, 384), confirm=(1152, 768), fresh=(1920, 80))


def candidates(count=512):
    rng = random.Random(20260924)
    choices = dict(topk=[8, 16, 24, 32, 48, 64, 96, 128],
                   eta=[640, 1280, 2560, 5120, 8192, 10240, 20480],
                   gamma=[0, .25, .5, 1., 2., 3.],
                   rel_cap=[.5, 1., 1.5, 2., 2.5, 3., 4.],
                   max_step=[.25, .5, 1., 1.5, 2., 3., 4.],
                   guidance_t_range=[(5, 45), (5, 35), (5, 25), (10, 45),
                                     (10, 35), (15, 45), (20, 45), (25, 45)])
    if not 1 <= count <= 512:
        raise ValueError("count must be between 1 and 512")
    def key(config):
        return tuple(tuple(config[k]) if k == 'guidance_t_range' else config[k]
                     for k in choices)
    seen = {key(REFERENCE), key(dict(REFERENCE, max_step=1.))}
    pool = []
    # Half the budget focuses on interactions near the incumbent; half is broad.
    while len(pool) < count:
        config = dict(REFERENCE)
        keys = list(choices) if len(pool) >= count // 2 else rng.sample(list(choices), 3)
        for k in keys:
            config[k] = rng.choice(choices[k])
        signature = key(config)
        if signature in seen:
            continue
        seen.add(signature)
        config['guidance_t_range'] = list(config['guidance_t_range'])
        config['name'] = f"broad_{len(pool):03d}"
        pool.append(config)
    return pool


def main():
    runner.EXTRA_BASELINES = [dict(runner.REFERENCE, name="previous_reference")]
    runner.REFERENCE = REFERENCE
    runner.SPLITS = SPLITS
    runner.SHORTLIST = 16
    runner.CAUTION = ("Screen/refine and 768-case confirmation reuse previously observed validation data; "
                      "confirmation is a reused audit, not an unbiased holdout. Only fresh indices "
                      "1920:2000 are unused in the documented search history; 80 cases have limited power. "
                      "No selection on fresh results. Seeds share samples. Test set untouched.")
    runner.candidates = candidates
    runner.main()


if __name__ == '__main__':
    main()