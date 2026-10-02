"""Round-2 validation search over axes unexplored by the round-1 fine search.

Round 1 (verbalts_fine_1013281) exhausted:
  - all 861 integer windows inside [5, 45]  -> no gain
  - eta in {3840..6400} x topk in {24..40}  -> no gain
  - max_step x rel_cap local refinement      -> no gain
and its audit found no candidate beating reference (test +1.70 pp historical).

Round 2 axes (everything else frozen at reference values
topk=32, eta=5120, gamma=0, window=[5,45], max_step=1., rel_cap=2.,
selection_score=applied, preserve_residual=True):
  gamma (adaptive confidence weighting), eta-up (7680..20480),
  active_only, iters (2/3), topk-down (8..20), selection_score=gradient,
  preserve_residual=False, per-head objective weights.

Screen uses the fresh validation window [1488, 2000), which was never used by
round 1; audit reuses the full 2000-sample validation like the round-1 audit.
"""
import argparse
import json
from pathlib import Path

from scripts.iterate_stabilized_steering import ROOT, write
from scripts import refine_stabilized_steering as scoring
from scripts.search_fine_verbalts import (
    REFERENCE, SEEDS, Search, digest, signature, provenance,
)

def candidates():
    pool, seen = [], set()

    def add(name, **overrides):
        config = dict(REFERENCE, name=name, **overrides)
        if signature(config) == signature(REFERENCE):
            return
        key = signature(config)
        if key in seen:
            return
        seen.add(key)
        pool.append(config)

    # A. gamma axis (adaptive confidence weighting), everything else = reference.
    for gamma in [.5, 1., 2., 3., 6., 9.]:
        add(f'g{gamma:g}', gamma=gamma)
    # B. eta-up x gamma (historical whole-curve champion used eta=10240, gamma=6).
    for eta in [7680, 10240, 15360, 20480]:
        for gamma in [3., 6.]:
            add(f'e{eta}_g{gamma:g}', eta=eta, gamma=gamma)
    # C. active_only x gamma x eta.
    for gamma in [0., 3., 6.]:
        for eta in [5120, 10240]:
            add(f'ao_g{gamma:g}_e{eta}', active_only=True, gamma=gamma, eta=eta)
    # D. iters x gamma.
    for iters in [2, 3]:
        for gamma in [0., 3., 6.]:
            add(f'i{iters}_g{gamma:g}', iters=iters, gamma=gamma)
    # E. topk-down x gamma.
    for topk in [8, 12, 16, 20]:
        for gamma in [0., 3., 6.]:
            add(f'k{topk}_g{gamma:g}', topk=topk, gamma=gamma)
    # F. score/residual switches.
    add('grad', selection_score='gradient')
    add('nores', preserve_residual=False)
    add('grad_nores', selection_score='gradient', preserve_residual=False)
    # G. per-head objective weights (graded B/M/E dose; ACCR needs all three).
    for weights in ([1., 2., 1.], [2., 1., 2.], [.5, 1., .5]):
        add('hw' + '_'.join(f'{w:g}'.replace('.', 'p') for w in weights),
            objective_head_weights=weights)
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=5.)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = provenance()
    path = out / 'provenance.json'
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError('source/data/checkpoint changed; use a new output directory')
    write(path, hashes)
    for source, expected_hash in hashes.items():
        source_path = Path(source)
        if source_path.suffix == '.py':
            snapshot = out / 'source' / source_path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source_path.read_bytes())
            if digest(snapshot) != expected_hash:
                raise ValueError('source changed during snapshot')
    pool = candidates()
    write(out / 'protocol.json', dict(
        reference=REFERENCE, candidates=pool, split='valid', seeds=list(SEEDS),
        batch_size=16, threshold=6,
        screen=dict(count=512, offset=1488,
                    note='fresh validation window unused by round 1'),
        audit=dict(count=2000, offset=0, seeds=[1, 7, 42, 123, 2026]),
        caution='Historically reused validation; no independent holdout; no test selection.',
        guardrails=dict(mean_mse=.01, per_seed_mse=.02, roughness=.02, curvature=.02)))
    scoring.PRIMARY_METRIC = 'whole_curve_exact_match'
    search = Search(out, args.hours)
    try:
        rows = search.stage('screen', pool, 512, 1488)
        frozen, seen = [], set()
        for row in sorted(rows, key=scoring.ranking_key):
            key = signature(row['config'])
            if row['eligible'] and key not in seen:
                frozen.append(row['config'])
                seen.add(key)
            if len(frozen) == 3:
                break
        write(out / 'frozen_final.json', [REFERENCE] + frozen)
        search.stage('audit', frozen or [dict(REFERENCE, name='unchanged')],
                     2000, 0, seeds=(1, 7, 42, 123, 2026))
        write(out / 'status.json', dict(state='completed'))
    except Exception as exc:
        write(out / 'status.json', dict(state='stopped', error=str(exc), resumable=True))
        raise


if __name__ == '__main__':
    main()
