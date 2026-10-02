"""Plot reproducibly selected successful and failed validation generations."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/weather_four_stage_128_morph_v2'
OUT = ROOT / 'results/weather128/valid_accr_1016853'


def main():
    samples = np.load(OUT / 'samples.npz')
    curves, labels, correct = [samples[k] for k in ('curves', 'labels', 'correct')]
    truth = np.load(DATA / 'valid_ts.npy')[:, :, 0]
    captions = np.load(DATA / 'valid_text_caps.npy')[:, 0]
    assert curves.shape == truth.shape == (4050, 128)
    assert correct.shape == labels.shape == (4050, 4)
    rng = np.random.default_rng(42)
    success = correct.all(axis=1)
    indices = np.concatenate([rng.choice(np.flatnonzero(success), 3, replace=False),
                              rng.choice(np.flatnonzero(~success), 3, replace=False)])
    names = {1: 'single peak', 2: 'double peaks', 3: 'sag'}
    records = []
    for group, selected in [('success', indices[:3]), ('failure', indices[3:])]:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        for ax, idx in zip(axes, selected):
            idx = int(idx)
            ax.plot(np.arange(1, 129), truth[idx], color='0.55', alpha=.8,
                    linewidth=1.4, label='Paired reference')
            ax.plot(np.arange(1, 129), curves[idx], color='#1769aa', linewidth=1.8,
                    label='VerbalTS generation')
            for stage in range(4):
                ok = bool(correct[idx, stage])
                ax.axvspan(stage * 32 + .5, (stage + 1) * 32 + .5,
                           color='#2ca02c' if ok else '#d62728', alpha=.07 if ok else .15)
                ax.text((stage + .5) / 4, .96,
                        f'Q{stage + 1}: {names[int(labels[idx, stage])]}\n'
                        f'{"OK" if ok else "WRONG"}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=10,
                        color='#196b22' if ok else '#b51f1f')
            for boundary in (32.5, 64.5, 96.5):
                ax.axvline(boundary, color='0.4', linestyle='--', linewidth=.8)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + .28 * (hi - lo))
            ax.set_xlim(.5, 128.5)
            ax.set_ylabel('Value (dataset scale)')
            ax.set_xlabel('Time point')
            ax.set_title(f'Validation index {idx} | {"ALL 4 CORRECT" if success[idx] else "NOT ALL CORRECT"}'
                         f' | paired MSE={samples["paired_mse"][idx]:.4f}', loc='left')
            ax.legend(loc='lower right', fontsize=9)
            ax.text(0, -.22, '\n'.join(textwrap.wrap(str(captions[idx]), 130)),
                    transform=ax.transAxes, fontsize=9, va='top')
            records.append(dict(index=idx, caption=str(captions[idx]),
                                labels=labels[idx].tolist(), correct=correct[idx].tolist(),
                                paired_mse=float(samples['paired_mse'][idx])))
        fig.suptitle(f'Weather128 pure VerbalTS | {group} cases | validation, sampling seed 42\n'
                     'Stage labels are targets; OK/WRONG is the CNN decision', fontsize=14)
        fig.subplots_adjust(top=.91, bottom=.08, hspace=.67)
        fig.savefig(OUT / f'cases_{group}.png', dpi=140)
        plt.close(fig)
    (OUT / 'cases.json').write_text(json.dumps(dict(selection_seed=42,
        selection='Random 3 successes and 3 failures, not representative proportions',
        cases=records), indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()