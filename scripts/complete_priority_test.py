"""Complete the user-requested three-seed test without waiting for validation search."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.iterate_stabilized_steering import ROOT, BASE, CKPT, write
from scripts.eval_frozen_full_test import TEST_SEEDS


def validate_report(report, configs, reference=None):
    if report['split'] != 'test' or sorted(report['indices']) != list(range(2000)):
        raise ValueError('Expected all 2000 test cases exactly once')
    if set(report['variants']) != {c['name'] for c in configs}:
        raise ValueError('Missing or unexpected variants')
    for config in configs:
        if report['variants'][config['name']]['config'] != config:
            raise ValueError('Configuration mismatch')
    if reference is not None:
        for key in ['indices', 'checkpoints', 'data_sha256', 'sampling', 'batch_size']:
            if report[key] != reference[key]:
                raise ValueError(f'Incompatible seed provenance: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'results/electricity_v3/quick_test_1011132'
    configs = json.loads((source / 'configs.json').read_text())
    reference = json.loads((source / 'seed42/summary.json').read_text())
    validate_report(reference, configs)
    if reference['seed'] != 42:
        raise ValueError('Expected seed42')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Refusing to overwrite test run')
    seeds = list(TEST_SEEDS)
    write(out / 'configs.json', configs)
    write(out / 'protocol.json', dict(split='test', all_samples=True, seeds=seeds, configs=configs,
        reused_seed42=str(source), caution='All original quick-test configurations retained. '
        'User corrected requested seeds to 1,7,42 after viewing seed42; replaces the earlier seed plans. '
        'No test-driven tuning or dropping variants.'))
    (out / 'seed42').symlink_to(source / 'seed42', target_is_directory=True)
    results = [dict(seed=42, n_samples=2000, variants=reference['variants'])]
    write(out / 'metrics.json', results)
    try:
        for seed in [s for s in seeds if s != 42]:
            write(out / 'status.json', dict(state='running', seed=seed, completed_seeds=[r['seed'] for r in results]))
            cmd = [sys.executable, '-m', 'sae.eval_morph_steering', '--data-root',
                str(ROOT / 'datasets/electricity_15min_semisynth_morph'), '--results-root', str(BASE),
                '--model-checkpoint', str(CKPT), '--output-dir', str(out / f'seed{seed}'),
                '--config-file', str(out / 'configs.json'), '--split', 'test', '--n-samples', '0',
                '--batch-size', '16', '--seed', str(seed), '--device', 'cuda', '--dynamic-threshold', '6']
            with (out / f'seed{seed}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((out / f'seed{seed}/summary.json').read_text())
            validate_report(report, configs, reference)
            if report['seed'] != seed:
                raise ValueError('Seed mismatch')
            results.append(dict(seed=seed, n_samples=2000, variants=report['variants']))
            write(out / 'metrics.json', results)
        write(out / 'status.json', dict(state='reporting', seeds=seeds))
        for candidate in ['reference', 'broad_060']:
            with (out / f'report_{candidate}.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'scripts.report_full_test_steering', '--run', str(out),
                    '--candidate', candidate], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(out / 'status.json', dict(state='completed', seeds=seeds))
    except Exception as exc:
        write(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()