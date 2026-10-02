"""Evaluate frozen validation-selected configurations once, without test selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.iterate_stabilized_steering import ROOT, BASE, CKPT, write

TEST_SEEDS = (1, 7, 42)


def validate_seeds(seeds):
    if len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    return list(seeds)


def frozen_configs(source):
    status = json.loads((source / 'status.json').read_text())
    if status.get('state') != 'completed':
        raise ValueError('Search must complete before test evaluation')
    frozen = json.loads((source / 'frozen_final.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    configs = [dict(name='pure'), dict(name='sae')] + protocol['extra_baselines'] + frozen
    if len({c['name'] for c in configs}) != len(configs):
        raise ValueError('Duplicate configuration names')
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TEST_SEEDS))
    parser.add_argument('--protocol-note', default='')
    args = parser.parse_args()
    seeds = validate_seeds(args.seeds)
    configs = frozen_configs(args.search_dir)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Refusing to overwrite test evaluation')
    write(out / 'configs.json', configs)
    write(out / 'protocol.json', dict(search=str(args.search_dir), split='test', all_samples=True,
          seeds=seeds, configs=configs, caution='Report every seed; no selection or tuning on test results. '
          + args.protocol_note))
    results = []
    try:
        for seed in seeds:
            write(out / 'status.json', dict(state='running', seed=seed))
            cmd = [sys.executable, '-m', 'sae.eval_morph_steering', '--data-root',
                   str(ROOT / 'datasets/electricity_15min_semisynth_morph'),
                   '--results-root', str(BASE), '--model-checkpoint', str(CKPT),
                   '--output-dir', str(out / f'seed{seed}'), '--config-file', str(out / 'configs.json'),
                   '--split', 'test', '--n-samples', '0', '--batch-size', '16',
                   '--seed', str(seed), '--device', 'cuda', '--dynamic-threshold', '6']
            with (out / f'seed{seed}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((out / f'seed{seed}/summary.json').read_text())
            from scripts.complete_priority_test import validate_report
            reference = (json.loads((out / f'seed{seeds[0]}/summary.json').read_text())
                         if results else None)
            validate_report(report, configs, reference)
            if report['seed'] != seed:
                raise ValueError('Seed mismatch')
            results.append(dict(seed=seed, n_samples=len(report['indices']), variants=report['variants']))
            write(out / 'metrics.json', results)
        write(out / 'status.json', dict(state='reporting', seeds=seeds))
        with (out / 'report.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'scripts.report_full_test_steering', '--run', str(out)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(out / 'status.json', dict(state='completed', seeds=seeds, report='REPORT.md'))
    except Exception as exc:
        write(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()