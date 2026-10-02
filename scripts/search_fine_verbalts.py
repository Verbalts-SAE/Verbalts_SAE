"""Resumable validation-only integer-window search; never evaluates test.

Each completed evaluation is committed with a hash manifest. Incomplete attempts
are retained, not overwritten. Validation has been reused historically.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.iterate_stabilized_steering import ROOT, BASE, CKPT, write
from scripts import refine_stabilized_steering as scoring

REFERENCE = dict(scoring.REFERENCE)
SEEDS = (1, 7, 42)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(config):
    return json.dumps({k: v for k, v in config.items() if k != 'name'}, sort_keys=True)


def windows():
    return [dict(REFERENCE, name=f'w{lo:02d}_{hi:02d}', guidance_t_range=[lo, hi])
            for lo in range(5, 46) for hi in range(lo, 46)]


def local_grid(centres, kind):
    axes = (dict(eta=[3840, 4480, 5120, 5760, 6400], topk=[24, 28, 32, 36, 40])
            if kind == 'strength' else
            dict(max_step=[.75, .875, 1., 1.125, 1.25], rel_cap=[1.5, 1.75, 2., 2.25, 2.5]))
    pool, seen = [], set()
    for centre in centres:
        for values in itertools.product(*axes.values()):
            config = dict(centre, **dict(zip(axes, values)))
            key = signature(config)
            if key not in seen:
                seen.add(key)
                pool.append(dict(config, name=f'{kind}_{len(pool):04d}'))
    return pool


def shortlist(rows, count=40):
    """Keep leaders and good representatives across window length/location bins."""
    selected = [r['config'] for r in rows[:count // 2]]
    bins = set()
    for row in rows:
        config = row['config']
        lo, hi = config['guidance_t_range']
        bucket = ((hi - lo) // 8, (lo + hi) // 16)
        if bucket not in bins and config not in selected:
            selected.append(config)
            bins.add(bucket)
        if len(selected) >= count:
            return selected[:count]
    for row in rows:
        if row['config'] not in selected:
            selected.append(row['config'])
        if len(selected) >= count:
            break
    return selected


def committed(report_path, marker):
    if not marker.exists() or not report_path.exists():
        return False
    files = json.loads(marker.read_text())
    return bool(files) and all((marker.parent / p).is_file() and
                              digest(marker.parent / p) == h for p, h in files.items())


class Search:
    def __init__(self, out, hours):
        self.out = out
        self.deadline = time.monotonic() + hours * 3600

    def evaluate(self, label, configs, count, offset, seed):
        folder = self.out / label
        folder.mkdir(parents=True, exist_ok=True)
        request = dict(configs=configs, count=count, offset=offset, seed=seed)
        request_path = folder / 'request.json'
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError(f'incompatible resume request: {label}')
        write(request_path, request)
        for attempt in sorted(folder.glob('attempt_*')):
            if committed(attempt / 'summary.json', attempt / 'complete.json'):
                return json.loads((attempt / 'summary.json').read_text())
        attempt = folder / f'attempt_{len(list(folder.glob("attempt_*"))):03d}'
        config_path = folder / 'configs.json'
        write(config_path, [dict(name='pure'), dict(name='sae'), REFERENCE] +
              [c for c in configs if signature(c) != signature(REFERENCE)])
        command = [sys.executable, '-m', 'sae.eval_morph_steering',
                   '--data-root', str(ROOT / 'datasets/electricity_15min_semisynth_morph'),
                   '--results-root', str(BASE), '--model-checkpoint', str(CKPT),
                   '--output-dir', str(attempt), '--config-file', str(config_path),
                   '--split', 'valid', '--n-samples', str(count), '--sample-offset', str(offset),
                   '--seed', str(seed), '--batch-size', '16', '--device', 'cuda',
                   '--dynamic-threshold', '6']
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('budget exhausted; completed evaluations can be resumed')
        write(self.out / 'status.json', dict(state='running', label=label, command=command))
        started = time.monotonic()
        with (folder / f'{attempt.name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining)
        report = json.loads((attempt / 'summary.json').read_text())
        expected = json.loads(config_path.read_text())
        if report['split'] != 'valid' or len(report['indices']) != count or any(
                c['name'] not in report['variants'] for c in expected):
            raise ValueError('incomplete evaluation')
        write(attempt / 'timing.json', dict(seconds=time.monotonic() - started))
        write(attempt / 'complete.json', {p.name: digest(p) for p in attempt.iterdir()
                                        if p.is_file() and p.name != 'complete.json'})
        return report

    def stage(self, name, pool, count, offset, seeds=SEEDS):
        rows = []
        # Batch candidates to amortize model loading and paired baseline generation.
        for start in range(0, len(pool), 24):
            group = pool[start:start + 24]
            reports = [self.evaluate(f'{name}_{start:04d}_s{s}', group, count, offset, s)
                       for s in seeds]
            for report in reports:
                for c in group:
                    if signature(c) == signature(REFERENCE):
                        report['variants'][c['name']] = report['variants']['reference']
            rows.extend(scoring.rank(reports, group))
            write(self.out / f'{name}_ranking.json', sorted(rows, key=scoring.ranking_key))
        return sorted(rows, key=scoring.ranking_key)


def provenance():
    paths = [CKPT, BASE / 'models/t5_45/best.pt',
             BASE / 'classwise_classifiers/t5_45/best.pt', BASE / 'cnn/segment_cnn.pth']
    paths += list((ROOT / 'sae').glob('*.py'))
    paths += list((ROOT / 'contsg/models').rglob('*.py'))
    paths += [Path(__file__), ROOT / 'scripts/refine_stabilized_steering.py',
              ROOT / 'scripts/iterate_stabilized_steering.py']
    data = ROOT / 'datasets/electricity_15min_semisynth_morph'
    paths += [data / f'valid_{suffix}.npy' for suffix in
              ('ts', 'attrs_idx', 'text_caps', 'cap_emb', 'tsfresh_global')]
    paths.append(data / 'train_tsfresh_global.npy')
    return {str(p.resolve()): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=23.)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = provenance()
    path = out / 'provenance.json'
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError('source/data/checkpoint changed; use a new output directory')
    write(path, hashes)
    # Preserve actual source, not only a dirty git revision or source hashes.
    for source, expected_hash in hashes.items():
        source_path = Path(source)
        if source_path.suffix == '.py':
            snapshot = out / 'source' / source_path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source_path.read_bytes())
            if digest(snapshot) != expected_hash:
                raise ValueError('source changed during snapshot')
    write(out / 'protocol.json', dict(reference=REFERENCE, window_count=861,
          split='valid', seeds=SEEDS, batch_size=16, threshold=6,
          caution='Historically reused validation; no independent holdout; no test selection.',
          guardrails=dict(mean_mse=.01, per_seed_mse=.02, roughness=.02, curvature=.02)))
    scoring.PRIMARY_METRIC = 'whole_curve_exact_match'
    search = Search(out, args.hours)
    try:
        if args.pilot:
            search.stage('pilot', [windows()[0], dict(REFERENCE, name='full'),
                                   dict(REFERENCE, name='middle', guidance_t_range=[20, 30])],
                         16, 0, seeds=(42,))
        else:
            rows = search.stage('screen', windows(), 256, 0)
            rows = search.stage('refine', shortlist(rows), 768, 256)
            centres = shortlist(rows, 8)
            strength = search.stage('strength', local_grid(centres, 'strength'), 512, 1024)
            caps = search.stage('caps', local_grid([r['config'] for r in strength[:12]], 'caps'),
                                512, 1024)
            combined = sorted(strength + caps, key=scoring.ranking_key)
            frozen, seen = [], set()
            for row in combined:
                key = signature(row['config'])
                if row['eligible'] and key not in seen:
                    frozen.append(row['config'])
                    seen.add(key)
                if len(frozen) == 3:
                    break
            write(out / 'frozen_final.json', [REFERENCE] + frozen)
            search.stage('audit', frozen or [dict(REFERENCE, name='unchanged')],
                         2000, 0, seeds=(1, 7, 42, 123, 2026))
        write(out / 'status.json', dict(state='completed', pilot=args.pilot))
    except Exception as exc:
        write(out / 'status.json', dict(state='stopped', error=str(exc), resumable=True))
        raise


if __name__ == '__main__':
    main()