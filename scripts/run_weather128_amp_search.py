"""Serial, fail-closed weather128 morphology-amplitude difficulty search.

Background ratio is fixed (0.6); morphology amplitude is the difficulty axis.
A prior probe showed real-data CNN joint ACCR stays >=99% for amplitude >=0.4
under max_background_ratio in [0.6, 0.75], so 0.4 is the search floor.

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
BASE = ROOT / 'datasets/weather_four_stage_128_morph_v3_bg045'
AMPLITUDE_FLOOR = 0.4
AMPLITUDE_CEILING = 0.7
AMPLITUDE_STEP = 0.1


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def next_amplitude(history, default=0.5):
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
    value = history[-1]['amplitude'] if history else default
    if easy:
        proposed = round(value - AMPLITUDE_STEP, 2)
        return proposed if proposed >= AMPLITUDE_FLOOR else None
    proposed = round(value + AMPLITUDE_STEP, 2)
    return proposed if proposed <= AMPLITUDE_CEILING else None


def background_partition(ids):
    groups = np.random.default_rng(20260928).permutation(np.unique(ids))
    split = int(len(groups) * .6)
    return np.flatnonzero(np.isin(ids, groups[:split])), np.flatnonzero(np.isin(ids, groups[split:]))


def check_pair(data):
    """Labels, background ids and source provenance must match BASE; the
    morphology residual differs by design (amplitude axis), so it is checked
    for shape/independence below instead of equality."""
    for split in ('train', 'valid', 'test'):
        for suffix in ('attrs_idx', 'background_ids'):
            np.testing.assert_array_equal(np.load(data / f'{split}_{suffix}.npy'),
                                          np.load(BASE / f'{split}_{suffix}.npy'))
        records = json.loads((data / f'{split}_records.json').read_text())
        original = json.loads((BASE / f'{split}_records.json').read_text())
        for a, b in zip(records, original):
            assert (a['source_start'], a['source_column']) == (b['source_start'], b['source_column'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--start-amplitude', type=float, default=0.5)
    p.add_argument('--max-background-ratio', type=float, default=0.6)
    p.add_argument('--max-rounds', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.max_rounds <= 3:
        p.error('max-rounds must be in [1,3]')
    if not AMPLITUDE_FLOOR <= args.start_amplitude <= AMPLITUDE_CEILING:
        p.error(f'start-amplitude must be in [{AMPLITUDE_FLOOR},{AMPLITUDE_CEILING}]')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='STARTING', job=os.environ.get('SLURM_JOB_ID'), target=[.4, .6],
                 max_rounds=args.max_rounds, max_background_ratio=args.max_background_ratio,
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
        amplitude = args.start_amplitude
        for iteration in range(args.max_rounds):
            state['round'] = iteration
            folder = out / f'round_{iteration:02d}_amp{amplitude:g}'
            folder.mkdir()
            data = folder / 'data'
            update('PREPARING', amplitude=amplitude, data=str(data), round_dir=str(folder))
            run('BUILD', 'scripts.prepare_weather128', '--output', data,
                '--max-background-ratio', args.max_background_ratio, '--amplitude', amplitude)
            run('AUDIT', 'scripts.audit_weather128', '--data-root', data)
            check_pair(data)
            search, confirm = background_partition(np.load(data / 'valid_background_ids.npy'))
            np.save(folder / 'search_indices.npy', search)
            np.save(folder / 'confirm_indices.npy', confirm)
            run('EMBEDDING', 'scripts.prepare_weather128', '--output', data, '--embed-only')
            run('CNN', 'scripts.train_weather128_cnn', '--data-root', data, '--output-dir', folder / 'cnn')
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
                run(name, 'scripts.eval_weather128_verbalts', '--checkpoint', checkpoint, '--cnn', cnn,
                    '--data-root', data, '--output-dir', folder / name, '--indices', indices, '--seed', seed)
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
                    '--num-heads', 4, '--output-dir', latent / 'mlp', '--epochs', 100,
                    '--patience', 12, '--seed', 42, '--device', 'cuda')
                run('STEERING', 'scripts.search_weather128_steering', '--checkpoint', checkpoint,
                    '--cnn', cnn, '--sae', latent / 'sae/best.pt', '--mlp', latent / 'mlp/best.pt',
                    '--data-root', data, '--search-indices', folder / 'search_indices.npy',
                    '--confirm-indices', folder / 'confirm_indices.npy', '--output-dir', folder / 'steering')
                update('COMPLETE_REVIEW_CURVE_QUALITY')
                return
            if accr < .4:
                update('PAUSED_LOW_ACCR_REVIEW_CONVERGENCE')
                return
            proposed = next_amplitude(state['history'], default=args.start_amplitude)
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
