"""Report all test seeds and predeclared fixed/improved/regressed visual cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.iterate_stabilized_steering import ROOT, write


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None)


def correct_matrix(predictions, targets):
    return np.asarray([[p['shape'] for p in row] for row in predictions]) == targets


def select_cases(pure_correct, steered_correct):
    p, s = pure_correct.all(1), steered_correct.all(1)
    groups = {'fixed': list(range(min(4, len(p)))),
              'fixed_by_steering': np.flatnonzero(~p & s)[:3].tolist(),
              'broken_by_steering': np.flatnonzero(p & ~s)[:3].tolist(),
              'both_incorrect': np.flatnonzero(~p & ~s)[:3].tolist(),
              'both_correct': np.flatnonzero(p & s)[:2].tolist()}
    return groups


def draw_case(path, curves, predictions, targets, title):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [3, 1.6]})
    colors = {'GT': 'black', 'VerbalTS': '#2878b5', 'Steering': '#e87522'}
    for name, curve in curves.items():
        axes[0].plot(curve, label=name, color=colors[name], lw=1.5, alpha=.85)
    for edge in [42.5, 84.5, 85.5]:
        axes[0].axvline(edge, color='gray', linestyle='--', alpha=.5)
    axes[0].axvspan(84.5, 85.5, color='gray', alpha=.2)
    axes[0].set_title(title, fontsize=10)
    axes[0].set_xlabel('Timestep; CNN windows: [0:43], [43:86], [85:128] (one-point overlap)')
    axes[0].legend()
    rows = [list(targets) + ['Target labels']]
    for name in curves:
        pred = predictions[name]
        matches = [p['shape'] == t for p, t in zip(pred, targets)]
        rows.append([f"{p['shape']}\nconf={p['joint_confidence']:.2f} {'OK' if ok else 'WRONG'}"
                     for p, ok in zip(pred, matches)] + [f'{sum(matches)}/3; all-correct={all(matches)}'])
    axes[1].axis('off')
    table = axes[1].table(cellText=rows, rowLabels=['Dataset target', 'CNN(GT)', 'CNN(VerbalTS)', 'CNN(Steering)'],
                          colLabels=['Beginning', 'Middle', 'End', 'Exact match'], loc='center',
                          colWidths=[.24, .24, .24, .28])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate', default=None, help='Explicit descriptive comparison, not test selection')
    args = parser.parse_args()
    out = args.run.resolve()
    protocol = json.loads((out / 'protocol.json').read_text())
    configs = protocol['configs']
    # Last frozen configuration is the validation-selected candidate, or reference if none qualified.
    candidate = args.candidate or configs[-1]['name']
    if candidate not in [c['name'] for c in configs]:
        raise ValueError('Unknown candidate')
    report_out = out if args.candidate is None else out / f'report_{candidate}'
    report_out.mkdir(exist_ok=True)
    seeds = protocol['seeds']
    reports = [json.loads((out / f'seed{s}/summary.json').read_text()) for s in seeds]
    if any(r['split'] != 'test' for r in reports):
        raise ValueError('Expected test results')
    if any(r['indices'] != reports[0]['indices'] for r in reports):
        raise ValueError('Seeds must share ordered test indices')
    indices = np.asarray(reports[0]['indices'])
    data = ROOT / 'datasets/electricity_15min_semisynth_morph'
    attrs = np.load(data / 'test_attrs_idx.npy')
    if len(indices) != len(attrs) or set(indices) != set(range(len(attrs))):
        raise ValueError('Expected complete test coverage')
    from sae.shapes import SHAPE_NAMES, classify_curves
    targets = np.asarray(SHAPE_NAMES)[attrs[indices]]
    truth = np.load(data / 'test_ts.npy')[indices, :, 0]
    captions = np.load(data / 'test_text_caps.npy', allow_pickle=True).reshape(-1)[indices]
    metrics = {'ACCR_percent': lambda v: 100 * v['cnn']['whole_curve_exact_match'],
               'segment_accuracy_percent': lambda v: 100 * v['cnn']['segment_accuracy'],
               'MSE': lambda v: v['mse']['overall_mse'],
               'roughness': lambda v: v['roughness'], 'curvature': lambda v: v['curvature']}
    aggregate = {}
    lines = ['# Full test steering report', '',
             f'Full test: {len(indices)} cases, seeds {seeds}. Candidate: `{candidate}`.',
             'ACCR = all three segments correct. Std is sample std across seeds (ddof=1).',
             'Seeds share test cases; seed std is not a confidence interval over new samples.',
             protocol.get('caution', ''), '',
             '| Variant | ACCR % (mean ± std) | MSE (mean ± std) | ACCR gain vs pure, pp (mean ± std) |',
             '|---|---|---|---|']
    for config in configs:
        name = config['name']
        values = {key: [fn(r['variants'][name]) for r in reports] for key, fn in metrics.items()}
        stats = {key: mean_std(value) for key, value in values.items()}
        deltas = {key: mean_std([fn(r['variants'][name]) - fn(r['variants']['pure']) for r in reports])
                  for key, fn in metrics.items()}
        aggregate[name] = dict(metrics=stats, paired_delta_vs_pure=deltas, per_seed=values)
        def fmt(v):
            return f"{v['mean']:.4f} ± {v['std']:.4f}" if v['std'] is not None else f"{v['mean']:.4f} (std unavailable)"
        lines.append(f"| {name} | {fmt(stats['ACCR_percent'])} | {fmt(stats['MSE'])} | {fmt(deltas['ACCR_percent'])} |")
    write(report_out / 'aggregate.json', aggregate)
    # Paired resampling of test cases, keeping every seed of a case together.
    correct = {}
    for config in configs:
        name = config['name']
        correct[name] = np.stack([correct_matrix(json.loads(
            (out / f'seed{s}/{name}_predictions.json').read_text()), targets).all(1) for s in seeds])
    intervals = {}
    for baseline in ['pure', 'reference', 'previous_reference']:
        if baseline not in correct:
            continue
        delta = (correct[candidate].astype(float) - correct[baseline]).mean(0)
        rng = np.random.default_rng(2026)
        boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
        intervals[baseline] = dict(mean_pp=float(delta.mean() * 100),
            ci95_pp=(np.quantile(boot, [.025, .975]) * 100).tolist())
    write(report_out / 'accr_paired_ci.json', intervals)
    lines += ['', '## ACCR paired 95% bootstrap intervals',
              f'Resample test cases, keeping their {len(seeds)} seeds together; conditional on these seeds.',
              '```json', json.dumps(intervals, indent=2), '```']
    # GT predictions use the exact frozen CNN checkpoint, never dataset labels as predictions.
    import torch
    from contsg.eval.metrics.segment import PeakValleyClassifier1D
    from sae.provenance import sha256_file
    checkpoint = next(Path(p) for p in reports[0]['checkpoints'] if p.endswith('/cnn/segment_cnn.pth'))
    if sha256_file(checkpoint) != reports[0]['checkpoints'][str(checkpoint)]:
        raise ValueError('CNN checkpoint changed since evaluation')
    cnn = PeakValleyClassifier1D(segment_len=43)
    cnn.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    gt_predictions = classify_curves(cnn, truth, torch.device('cpu'))
    write(out / 'gt_predictions.json', gt_predictions)
    gt_correct = correct_matrix(gt_predictions, targets)
    lines += ['', f'CNN(GT) ACCR: {gt_correct.all(1).mean():.2%}; segment accuracy: {gt_correct.mean():.2%}.',
              '', '## Cases (fixed seed42)',
              'Fixed first four positions plus first cases in each outcome group. These are illustrative, not an unbiased subset.',
              'Confidence is the product of peak/valley head probabilities, not calibrated certainty.',
              'Types: nothing = no target local shape; single peak; double peaks; sag = valley.', '']
    folder = out / 'seed42'
    pp, sp = [json.loads((folder / f'{name}_predictions.json').read_text()) for name in ['pure', candidate]]
    pure, steered = [np.load(folder / f'{name}.npy') for name in ['pure', candidate]]
    groups = select_cases(correct_matrix(pp, targets), correct_matrix(sp, targets))
    case_dir = report_out / 'cases'
    case_dir.mkdir(exist_ok=True)
    manifest = []
    for group, positions in groups.items():
        if not positions:
            lines.append(f'- {group}: no cases.')
        for pos in positions:
            filename = f'{group}_test{indices[pos]}.png'
            pm = float(np.mean((pure[pos] - truth[pos]) ** 2))
            sm = float(np.mean((steered[pos] - truth[pos]) ** 2))
            draw_case(case_dir / filename, dict(GT=truth[pos], VerbalTS=pure[pos], Steering=steered[pos]),
                      dict(GT=gt_predictions[pos], VerbalTS=pp[pos], Steering=sp[pos]), targets[pos],
                      f'Test {indices[pos]} | {group} | seed42 | {candidate}\nMSE: VerbalTS={pm:.4f}, Steering={sm:.4f}')
            manifest.append(dict(group=group, position=pos, test_index=int(indices[pos]),
                caption=str(captions[pos]), targets=targets[pos].tolist(),
                predictions=dict(GT=gt_predictions[pos], VerbalTS=pp[pos], Steering=sp[pos]),
                mse=dict(VerbalTS=pm, Steering=sm), file=filename))
            lines.append(f'- [{group}: test {indices[pos]}](cases/{filename})')
    write(case_dir / 'manifest.json', manifest)
    (report_out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()