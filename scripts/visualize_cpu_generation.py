"""Reproducible previews of saved CPU test generations, without resampling."""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('nothing', 'single peak', 'double peaks', 'sag', 'invalid combination')


def main():
    data = ROOT / 'datasets/cpu_shape_v1'
    pipeline = ROOT / 'results/cpu_shape_v1/pipeline'
    out = pipeline / 'visualizations'
    out.mkdir(parents=True, exist_ok=True)
    truth = np.load(data / 'test_ts.npy', allow_pickle=False)[..., 0]
    labels = np.load(data / 'test_attrs_idx.npy', allow_pickle=False)
    captions = np.load(data / 'test_text_caps.npy', allow_pickle=False).reshape(-1)
    with np.load(pipeline / 'generated_seed1.npz', allow_pickle=False) as saved:
        curves, predicted = saved['curves'], saved['predicted']
        np.testing.assert_array_equal(saved['labels'], labels)
    assert curves.shape == truth.shape == (len(labels), 128)
    assert np.isfinite(curves).all() and np.isfinite(truth).all()
    correct = (predicted == labels).all(1)
    rng = np.random.default_rng(42)
    # One unfiltered random example per class, then explicit successes/failures.
    classes = labels.max(1)
    groups = {'class_random': [int(rng.choice(np.flatnonzero(classes == k))) for k in range(4)]}
    used = set(groups['class_random'])
    for name, mask in (('success', correct), ('failure', ~correct)):
        pool = np.array([i for i in np.flatnonzero(mask) if i not in used])
        groups[name] = rng.choice(pool, size=min(4, len(pool)), replace=False).tolist()
        used.update(groups[name])

    def label_text(values):
        return ' / '.join(NAMES[int(k)] for k in values)

    def draw(ax, i):
        ax.plot(truth[i], color='#3274a1', lw=1.5, label='Real reference')
        ax.plot(curves[i], color='#e1812c', lw=1.5, label='VerbalTS generated (seed 1)')
        ax.axvline(42.5, color='gray', ls=':', alpha=.65)
        ax.axvline(85.5, color='gray', ls=':', alpha=.65)
        ax.set_title(f'Test #{i} | {"PASS" if correct[i] else "FAIL"} (CNN local labels)\n'
                     f'Target B/M/E: {label_text(labels[i])}\n'
                     f'CNN B/M/E: {label_text(predicted[i])}', fontsize=9)
        ax.set_xlim(0, 127)
        ax.set_xlabel('Time index (sampling interval unknown)')
        ax.set_ylabel('Dataset value (no extra normalization)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    entries = []
    lines = ['# CPU VerbalTS generation cases', '',
             'Existing test generations, seed 1; selection RNG seed 42. No regeneration or smoothing.',
             'Blue: real reference; orange: generated. Matching a caption does not require pointwise reconstruction.',
             'PASS/FAIL uses only CNN local morphology; it does not assess the background caption.',
             'B/M/E CNN windows: [0:43], [43:86], [85:128] (one-point overlap).', '']
    for group, ids in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, i in zip(axes.flat, ids):
            ax.set_visible(True)
            draw(ax, i)
        fig.suptitle(f'CPU generation | {group} | shared raw scale within each case')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out / f'{group}.png', dpi=150)
        plt.close(fig)
        lines += [f'## {group}', '', f'![{group}]({group}.png)', '']
        for i in ids:
            fig, ax = plt.subplots(figsize=(12, 6))
            draw(ax, i)
            fig.text(.06, .025, textwrap.fill(str(captions[i]), width=135), fontsize=9, va='bottom')
            fig.tight_layout(rect=(0, .24, 1, 1))
            filename = f'case_{i:04d}.png'
            fig.savefig(out / filename, dpi=150)
            plt.close(fig)
            entries.append(dict(index=i, group=group, correct=bool(correct[i]),
                                target=labels[i].tolist(), predicted=predicted[i].tolist(),
                                caption=str(captions[i]), image=filename))
            lines += [f'### [Case {i}]({filename})', str(captions[i]), '']
    (out / 'manifest.json').write_text(json.dumps(dict(generation_seed=1, selection_seed=42,
        selection='one random per class (unfiltered), four random successes, four random failures; no duplicates',
        source=str(pipeline / 'generated_seed1.npz'), cases=entries), indent=2) + '\n')
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(groups))
    print(out)


if __name__ == '__main__':
    main()