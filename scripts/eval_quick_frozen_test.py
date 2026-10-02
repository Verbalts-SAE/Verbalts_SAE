"""One predeclared candidate, full test seed42; never feed results into search."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.iterate_stabilized_steering import ROOT, BASE, CKPT, write


def quick_configs():
    source = ROOT / 'results/electricity_v3/large_search_1011120/screen_060.json'
    configs = json.loads(source.read_text())
    names = ['pure', 'sae', 'reference', 'previous_reference', 'broad_060']
    return [next(c for c in configs if c['name'] == name) for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Refusing to overwrite a frozen test run')
    configs = quick_configs()
    write(out / 'configs.json', configs)
    write(out / 'protocol.json', dict(split='test', all_samples=True, seeds=[42], configs=configs,
          caution='Predeclared broad_060 from validation screen; single-seed descriptive early test. '
                  'No seed std available. Do not use these results to change search or select candidates.'))
    try:
        write(out / 'status.json', dict(state='running', seed=42))
        command = [sys.executable, '-m', 'sae.eval_morph_steering', '--data-root',
                   str(ROOT / 'datasets/electricity_15min_semisynth_morph'),
                   '--results-root', str(BASE), '--model-checkpoint', str(CKPT),
                   '--output-dir', str(out / 'seed42'), '--config-file', str(out / 'configs.json'),
                   '--split', 'test', '--n-samples', '0', '--batch-size', '16',
                   '--seed', '42', '--device', 'cuda', '--dynamic-threshold', '6']
        with (out / 'seed42.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((out / 'seed42/summary.json').read_text())
        write(out / 'metrics.json', [dict(seed=42, n_samples=len(report['indices']), variants=report['variants'])])
        write(out / 'status.json', dict(state='reporting', seed=42))
        with (out / 'report.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'scripts.report_full_test_steering', '--run', str(out)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(out / 'status.json', dict(state='completed', seeds=[42], report='REPORT.md'))
    except Exception as exc:
        write(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()