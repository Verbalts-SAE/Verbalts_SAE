"""Serial, fail-closed weather128 five-class fine-label difficulty search.

Background ratio is fixed (0.6); morphology amplitude is the difficulty axis.
The label space grows to 5^4 = 625 combinations per background, and captions
carry fine-grained morphology plus background descriptions.

A prior probe showed the 5-class CNN reaches 99.99% joint ACCR on real data
at amplitude 1.2, so the CNN gate stays >=0.99.

Never scores test data for difficulty or steering selection.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
AMPLITUDES = (1.2, 0.8, 0.5, 0.35)
MAX_BACKGROUND_RATIO = 0.6
TARGET = (.4, .6)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def next_amplitude(history):
    """Bracket on observed results; bounded search, not a monotonicity claim."""
    easy = [r['amplitude'] for r in history if r['accr'] > .6]
    hard = [r['amplitude'] for r in history if r['accr'] < .4]
    if history and .4 <= history[-1]['accr'] <= .6:
        return None
    if easy and hard:
        lo, hi = max(hard), min(easy)
        if lo >= hi:
            return None
        return round((lo + hi) / 2, 2)
    value = history[-1]['amplitude'] if history else AMPLITUDES[0]
    if easy:
        lower = [a for a in AMPLITUDES if a < value - 1e-9]
        return max(lower) if lower else None
    higher = [a for a in AMPLITUDES if a > value + 1e-9]
    return min(higher) if higher else None


def background_partition(ids, sample_size):
    rng = np.random.default_rng(20260928)
    groups = rng.permutation(np.unique(ids))
    split = int(len(groups) * .6)
    search, confirm = groups[:split], groups[split:]
    pick = lambda mask: rng.choice(np.flatnonzero(mask), size=min(sample_size, int(mask.sum())),
                                   replace=False)
    return pick(np.isin(ids, search)), pick(np.isin(ids, confirm))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-background-ratio', type=float, default=MAX_BACKGROUND_RATIO)
    p.add_argument('--sample-size', type=int, default=3000,
                   help='Validation samples scored per phase (background-disjoint partition)')
    args = p.parse_args()
    if args.sample_size < 1000:
        p.error('sample-size must be at least 1000 for a stable ACCR estimate')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='STARTING', job=os.environ.get('SLURM_JOB_ID'), target=list(TARGET),
                 amplitudes=list(AMPLITUDES), max_background_ratio=args.max_background_ratio,
                 history=[], test_evaluated=False,
                 confirmation_note='Disjoint from difficulty search; used in model validation.')

    def update(status, **extra):
        state.update(status=status, updated=dt.datetime.now().astimezone().isoformat(), **extra)
        save(out / 'status.json', state)
        print(state['updated'], status, extra, flush=True)

    def run(stage, module, *argv):
        command = [sys.executable, '-u', '-m', module, *map(str, argv)]
        log = out / f'{state.get("round", 0):02d}_{stage}.log'
        update(stage, command=command, log=str(log))
        with log.open('w') as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    tail = log.read_text(errors='replace')[-4000:]
                    save(out / 'monitor.json', dict(time=dt.datetime.now().astimezone().isoformat(),
                         stage=stage, pid=process.pid, log=str(log), tail=tail,
                         log_age_seconds=time.time() - log.stat().st_mtime))
            if code:
                raise RuntimeError(f'{stage} failed with exit code {code}; see {log}')

    try:
        amplitude = AMPLITUDES[0]
        for iteration in range(len(AMPLITUDES)):
            state['round'] = iteration
            folder = out / f'round_{iteration:02d}_amp{amplitude:g}'
            folder.mkdir()
            data = folder / 'data'
            update('PREPARING', amplitude=amplitude, data=str(data), round_dir=str(folder))
            run('BUILD', 'scripts.prepare_weather128_fine', '--output', data,
                '--max-background-ratio', args.max_background_ratio, '--amplitude', amplitude)
            run('AUDIT', 'scripts.audit_weather128_fine', '--data-root', data)
            audit = json.loads((data / 'audit.json').read_text())
            assert audit['status'] == 'PASS'
            search, confirm = background_partition(np.load(data / 'valid_background_ids.npy'),
                                                   args.sample_size)
            np.save(folder / 'search_indices.npy', search)
            np.save(folder / 'confirm_indices.npy', confirm)
            run('EMBEDDING', 'scripts.prepare_weather128_fine', '--output', data, '--embed-only')
            run('CNN', 'scripts.train_weather128_fine_cnn', '--data-root', data,
                '--output-dir', folder / 'cnn')
            cnn_report = json.loads((folder / 'cnn/report.json').read_text())
            if cnn_report['best_accr'] < .99:
                update('PAUSED_CNN_RELIABILITY', cnn_accr=cnn_report['best_accr'])
                return
            cfg = yaml.safe_load((ROOT / 'results/weather128/configs/verbalts_lr0.0003.yaml').read_text())
            cfg['data']['data_folder'] = str(data)
            cfg['eval']['cache_folder'] = str(folder / 'cache')
            from contsg.config.schema import ExperimentConfig
            ExperimentConfig(**cfg)
            config = folder / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            run('VERBALTS', 'contsg.cli', 'train', '--config', config, '--output-dir', folder / 'experiments',
                '--no-eval', '--progress', 'log')
            from sae.resolve_best_checkpoint import resolve_best_checkpoint
            summaries = list((folder / 'experiments').glob('*/summary.json'))
            if len(summaries) != 1:
                raise RuntimeError(f'Expected one completed training summary: {summaries}')
            checkpoint = resolve_best_checkpoint(summaries[0].parent)
            cnn = folder / 'cnn/segment_cnn.pth'

            def evaluate(name, indices, seed):
                run(name, 'scripts.eval_weather128_fine_verbalts', '--checkpoint', checkpoint,
                    '--cnn', cnn, '--data-root', data, '--output-dir', folder / name,
                    '--indices', indices, '--seed', seed)
                return json.loads((folder / name / 'report.json').read_text())

            result = evaluate('PURE_SEARCH', folder / 'search_indices.npy', 42)
            accr = result['cnn']['accr']
            state['history'].append(dict(amplitude=amplitude, accr=accr, folder=str(folder),
                                         checkpoint=str(checkpoint)))
            update('BASELINE_EVALUATED', accr=accr)
            if .4 <= accr <= .6:
                confirmed = [evaluate(f'PURE_CONFIRM_{s}', folder / 'confirm_indices.npy', s)['cnn']['accr']
                             for s in (1, 7, 42)]
                update('TARGET_CANDIDATE', confirm_accr=confirmed, confirm_mean=float(np.mean(confirmed)))
                if not .4 <= np.mean(confirmed) <= .6:
                    update('PAUSED_CONFIRMATION_OUTSIDE_TARGET')
                    return
                latent = folder / 'latents'
                for split in ('train', 'valid'):
                    run(f'ACTIVATIONS_{split}', 'sae.activations', '--ckpt-path', checkpoint,
                        '--data-root', data, '--split', split, '--target-t-ranges', '5-45',
                        '--output-dir', latent / 'activations', '--batch-size', 64, '--device', 'cuda')
                train_cache = latent / 'activations/train_layer1_t5_45.pt'
                valid_cache = latent / 'activations/valid_layer1_t5_45.pt'
                run('SAE', 'sae.training.train_sae', '--activations', train_cache,
                    '--valid-activations', valid_cache, '--output-dir', latent / 'sae',
                    '--t-range', 5, 45, '--input-dim', 64, '--latent-dim', 512, '--topk-k', 48,
                    '--aux-lambda', .01, '--dead-tolerance', 500, '--lr', .0001,
                    '--batch-size', 4096, '--epochs', 100, '--warmup-steps', 500,
                    '--eval-interval', 10, '--seed', 42, '--device', 'cuda')
                run('MLP', 'sae.train_classifier', '--sae-checkpoint', latent / 'sae/best.pt',
                    '--train-cache', train_cache, '--valid-cache', valid_cache,
                    '--train-attrs', data / 'train_attrs_idx.npy', '--valid-attrs', data / 'valid_attrs_idx.npy',
                    '--num-heads', 4, '--num-shape-classes', 6, '--output-dir', latent / 'mlp',
                    '--epochs', 100, '--patience', 12, '--seed', 42, '--device', 'cuda')
                run('STEERING', 'scripts.search_weather128_fine_steering', '--checkpoint', checkpoint,
                    '--cnn', cnn, '--sae', latent / 'sae/best.pt', '--mlp', latent / 'mlp/best.pt',
                    '--data-root', data, '--search-indices', folder / 'search_indices.npy',
                    '--confirm-indices', folder / 'confirm_indices.npy', '--output-dir', folder / 'steering')
                update('COMPLETE_REVIEW_CURVE_QUALITY')
                return
            if accr < .4:
                update('PAUSED_LOW_ACCR_REVIEW_CONVERGENCE')
                return
            proposed = next_amplitude(state['history'])
            if proposed is None or proposed in [r['amplitude'] for r in state['history']]:
                update('PAUSED_NO_NEW_DIFFICULTY')
                return
            amplitude = proposed
        update('ROUND_LIMIT_REACHED')
    except BaseException as exc:
        update('FAILED', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
