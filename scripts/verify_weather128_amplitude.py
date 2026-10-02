"""Probe CNN real-data reliability at reduced morphology amplitudes.

Independent of the main difficulty search: builds small probe datasets with
the same background construction (max_background_ratio) but weaker morphology
amplitude, then trains the frozen CNN and reports its real-data joint ACCR.

The CNN must stay >=99% on real data for a candidate amplitude to be usable
in the main 40%-60% difficulty search.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--amplitudes', type=float, nargs='+', default=[0.7, 0.6, 0.5])
    p.add_argument('--max-background-ratio', type=float, default=0.75)
    p.add_argument('--counts', type=int, nargs=3, default=(200, 50, 50))
    p.add_argument('--seed', type=int, default=20260928)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(max_background_ratio=args.max_background_ratio, counts=list(args.counts),
                  seed=args.seed, results=[])

    from scripts.prepare_weather128 import prepare, SOURCE
    for amp in args.amplitudes:
        data = out / f'amp{amp:g}'
        prepare(SOURCE, data, counts=tuple(args.counts), seed=args.seed, amplitude=amp,
                max_background_ratio=args.max_background_ratio)
        cnn_dir = data / 'cnn'
        log = out / f'cnn_amp{amp:g}.log'
        with log.open('w') as handle:
            proc = subprocess.run([sys.executable, '-u', '-m', 'scripts.train_weather128_cnn',
                                   '--data-root', data, '--output-dir', cnn_dir],
                                  cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'CNN failed for amplitude={amp}; see {log}')
        accr = json.loads((cnn_dir / 'report.json').read_text())['best_accr']
        report['results'].append(dict(amplitude=amp, cnn_real_valid_accr=accr))
        (out / 'probe_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'amplitude={amp:g} cnn_real_valid_accr={accr:.4f}', flush=True)


if __name__ == '__main__':
    main()
