"""Visualize paired pulse/window experiments without running generation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sae.shapes import SHAPE_NAMES
from sae.provenance import write_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    reports = {n: json.loads((args.run / f'length{n}/summary.json').read_text()) for n in (1, 5)}
    indices = reports[1]['indices']
    assert indices == reports[5]['indices']
    for key in ('seed', 'batch_size', 'checkpoints', 'data_sha256', 'steps'):
        assert reports[1][key] == reports[5][key], key
    for name in ('valid_ts.npy', 'valid_attrs_idx.npy'):
        assert sha256_file(args.data_root / name) == reports[1]['data_sha256'][name]
    truth = np.load(args.data_root / 'valid_ts.npy')[indices, :, 0]
    targets = np.asarray(SHAPE_NAMES)[np.load(args.data_root / 'valid_attrs_idx.npy')[indices]]
    baseline = np.load(args.run / 'length1/sae.npy')
    np.testing.assert_array_equal(baseline, np.load(args.run / 'length5/sae.npy'))
    out = args.run / 'figures'
    out.mkdir(exist_ok=True)
    files = []

    def save(fig, name):
        fig.savefig(out / name, dpi=160, bbox_inches='tight')
        plt.close(fig)
        files.append(name)

    steps = [40, 25, 10]
    labels = [f't={t}\n{v}' for t in steps for v in ('reference', 'active_only')]
    keys = [f't{t}_{v}' for t in steps for v in ('reference', 'active_only')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for n, offset, color in [(1, -.18, '#377eb8'), (5, .18, '#e66101')]:
        rows = [reports[n]['results'][k] for k in keys]
        axes[0].bar(np.arange(6) + offset, [np.mean(r['final_rmse']) for r in rows],
                    width=.34, label=f'{n} step(s)', color=color)
        axes[1].bar(np.arange(6) + offset,
                    [100 * r['probability']['mean_delta_per_stage'][1] for r in rows],
                    width=.34, label=f'{n} step(s)', color=color)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Mean per-sample final RMSE vs SAE (log scale)')
    axes[0].set_title('Larger response does not imply better morphology')
    axes[1].set_ylabel('Middle target probability change (percentage points)')
    axes[1].set_title('Middle-stage response remains small')
    axes[1].axhline(0, color='black', lw=.7)
    for ax in axes:
        ax.set_xticks(range(6), labels, fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Paired validation batch (N=16), eta=5120; SAE-only after intervention')
    fig.tight_layout()
    save(fig, '01_response_overview.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    matrices = [np.array([reports[5]['results'][f't40_{v}']['probability']['target_probability_delta'][i]
                          for i in range(len(indices))]) * 100 for v in ('reference', 'active_only')]
    limit = max(np.abs(m).max() for m in matrices)
    for ax, matrix, variant in zip(axes, matrices, ('reference', 'active_only')):
        im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(3), ['Beginning', 'Middle', 'End'])
        ax.set_yticks(range(len(indices)), indices)
        ax.set_title(f'{variant}: t=40..36')
        for i in range(len(indices)):
            for j in range(3):
                ax.text(j, i, f'{matrix[i, j]:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(matrix[i, j]) > .55 * limit else 'black')
    axes[0].set_ylabel('Validation sample index (original batch order)')
    fig.colorbar(im, ax=axes, label='Target probability change (percentage points)', shrink=.8)
    fig.suptitle('Per-sample direction: target peak probability × target valley probability\n'
                 'CNN probabilities can saturate; near-zero change is not proof of no shape change')
    save(fig, '02_target_probability_heatmap.png')

    data = {}
    for n in (1, 5):
        for v in ('reference', 'active_only'):
            for t in steps:
                with np.load(args.run / f'length{n}/t{t}_{v}.npz') as z:
                    data[n, v, t] = z['final'].copy()
    # Two pre-specified diagnostic cases, not a representative success sample.
    for idx in (1493, 299):
        i = indices.index(idx)
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for n, ax in zip((1, 5), axes[:2]):
            ax.plot(truth[i], color='.65', lw=1.3, label='Ground truth (one valid realization)')
            ax.plot(baseline[i], color='black', lw=1.6, label='SAE-only')
            for v, color in [('reference', '#377eb8'), ('active_only', '#e66101')]:
                ax.plot(data[n, v, 40][i], color=color, lw=1.3, linestyle='--' if v == 'reference' else '-',
                        label=f'{v}, {n} step(s)')
            pred = reports[n]['results']['t40_active_only']['predictions'][i]
            ax.set_title(f'{n} step(s), start t=40 | active_only CNN: ' + ' / '.join(p['shape'] for p in pred), fontsize=10)
            ax.set_ylabel('Model-scale value')
            ax.legend(fontsize=8, ncol=2)
        lo = min(ax.get_ylim()[0] for ax in axes[:2]); hi = max(ax.get_ylim()[1] for ax in axes[:2])
        for ax in axes[:2]:
            ax.set_ylim(lo, hi)
        for n, style in [(1, '--'), (5, '-')]:
            for v, color in [('reference', '#377eb8'), ('active_only', '#e66101')]:
                axes[2].plot(data[n, v, 40][i] - baseline[i], style, color=color,
                             label=f'{v}, {n} step(s)')
        axes[2].axhline(0, color='black', lw=.7)
        axes[2].set_ylabel('Steered − SAE-only\n(separate difference scale)')
        axes[2].legend(fontsize=8, ncol=2)
        for ax in axes:
            ax.axvspan(43, 85, color='orange', alpha=.07)
            for x in (42.5, 85.5):
                ax.axvline(x, color='.5', linestyle=':')
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('Time index (no per-curve normalization)')
        reason = 'Persistent middle/end failure' if idx == 1493 else 'Only observed fix: beginning, nothing → sag'
        fig.suptitle(f'Validation {idx}: {reason}\nTarget: ' + ' / '.join(targets[i]))
        fig.tight_layout(rect=(0, 0, 1, .94))
        save(fig, f'case_{idx}_pulse_vs_window.png')

    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(truth[i], color='.7', lw=.8)
        ax.plot(baseline[i], color='black', lw=1)
        ax.plot(data[5, 'active_only', 40][i], color='#e66101', lw=1)
        ax.axvspan(43, 85, color='orange', alpha=.07)
        ax.set_title(f'{indices[i]} | ' + '/'.join(targets[i]), fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('All 16 samples (not selected): active_only, t=40..36\n'
                 'Gray: ground truth | Black: SAE-only | Orange: steered; panel y-scales differ')
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, '05_all_samples.png')
    write_json(out / 'manifest.json', dict(files=files, indices=indices, case_selection={
        '1493': 'previously tracked persistent failure', '299': 'only fixed stage in this experiment'},
        summaries={str(n): sha256_file(args.run / f'length{n}/summary.json') for n in (1, 5)},
        plotting_source_sha256=sha256_file(Path(__file__))))
    print(f'Saved {len(files)} figures to {out}', flush=True)


if __name__ == '__main__':
    main()