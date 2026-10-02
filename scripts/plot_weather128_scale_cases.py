"""Paired validation previews changing only the weather background cap."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/weather_four_stage_128_morph_v2'
OUT = ROOT / 'results/weather128/background_scale_preview'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratios', type=float, nargs=3, default=(.3, .6, 1.))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--raw-preview', action='store_true',
                        help='Compare baseline, unscaled centered background, and raw absolute background')
    args = parser.parse_args()
    ratios = tuple(args.ratios)
    if ratios[0] != .3 or not np.isfinite(ratios).all() or not all(
            a < b for a, b in zip(ratios, ratios[1:])):
        parser.error('Ratios must start at 0.3 and be finite and strictly increasing')
    out = args.output
    meta = json.loads((DATA / 'meta.json').read_text())
    records = json.loads((DATA / 'valid_records.json').read_text())
    bg = np.load(DATA / 'valid_backgrounds.npy')
    ids = np.load(DATA / 'valid_background_ids.npy')
    labels = np.load(DATA / 'valid_attrs_idx.npy')
    curves = np.load(DATA / 'valid_ts.npy')[:, :, 0]
    amplitude = meta['amplitude']
    assert meta['max_background_ratio'] == .3
    old_scale = np.array([r['scale'] for r in records])
    assert np.all(old_scale > 0)
    raw = bg / old_scale[:, None]
    span = np.ptp(raw, axis=1)
    steps = np.abs(np.diff(raw, axis=1))
    gentle = (span == 0) | ((steps.max(1) <= .25 * span) & (steps.sum(1) <= 2.5 * span))
    base_scale = np.array([1. if r['channel'] == 'air_pressure' else (.4 if g else .2)
                           for r, g in zip(records, gentle)])
    stage_span = np.ptp(raw.reshape(-1, 4, 32), axis=2).max(1)
    scales = np.stack([np.minimum(base_scale, amplitude * ratio /
                                  np.maximum(stage_span, 1e-30)) for ratio in ratios])
    np.testing.assert_allclose(scales[0], old_scale, atol=1e-12)
    all_bg = raw[None] * scales[:, :, None]
    np.testing.assert_allclose(all_bg[0], bg, atol=1e-12)
    if args.raw_preview:
        first = np.array([r['raw_first'] for r in records])
        all_bg = np.stack((bg, raw, raw + first[:, None]))
        scales = np.stack((old_scale, np.ones_like(old_scale), np.ones_like(old_scale)))
    out.mkdir(parents=True, exist_ok=True)
    names = {1: 'single peak', 2: 'double peaks', 3: 'sag'}
    targets = ((1, 2, 3, 1), (2, 3, 1, 2))
    saved, arrays = [], []
    for channel in ('temperature', 'wind_speed', 'air_pressure'):
        candidates = [i for i, r in enumerate(records) if r['channel'] == channel]
        ranked = sorted(candidates, key=lambda i: (stage_span[i] * base_scale[i], i))
        selected = (ranked[len(ranked) // 2], ranked[-1])
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True,
                                 sharey=False if args.raw_preview else 'row')
        for row, (bid, target) in enumerate(zip(selected, targets)):
            matches = np.flatnonzero((ids == bid) & (labels == target).all(1))
            assert len(matches) == 1
            idx = int(matches[0])
            residual = curves[idx].astype(float) - bg[bid]
            paired = all_bg[:, bid] + residual
            assert np.isfinite(paired).all()
            np.testing.assert_allclose(paired[0], curves[idx], atol=1e-12)
            np.testing.assert_allclose(paired - all_bg[:, bid],
                                       np.broadcast_to(residual, paired.shape), atol=1e-12)
            arrays.append(paired)
            if args.raw_preview:
                lo = min(paired[:2].min(), all_bg[:2, bid].min())
                hi = max(paired[:2].max(), all_bg[:2, bid].max())
                pad = max((hi - lo) * .05, .01)
                for ax in axes[row, :2]:
                    ax.set_ylim(lo - pad, hi + pad)
                axes[row, 2].set_ylim(lo - pad + first[bid], hi + pad + first[bid])
                np.testing.assert_allclose(paired[2] - first[bid], paired[1], atol=1e-12)
            for col, ratio in enumerate(ratios):
                ax = axes[row, col]
                ax.plot(all_bg[col, bid], '--', color='0.45', linewidth=1.4, label='Weather background')
                ax.plot(paired[col], color=('#1769aa', '#e48b16', '#b33752')[col],
                        linewidth=1.3, label='Background + fixed morphology/noise')
                for boundary in (31.5, 63.5, 95.5):
                    ax.axvline(boundary, color='0.7', linewidth=.8)
                title = (('Baseline cap 0.30', 'Raw minus first; scale=1', 'Raw absolute; no transform')[col]
                         if args.raw_preview else f'Cap ratio {ratio:.2f}')
                ax.set_title(f'{title} | background x{scales[col, bid]/old_scale[bid]:.2f}\n'
                             f'valid index {idx}, background {bid}', fontsize=10)
                ax.set_xlim(0, 127)
                ax.grid(alpha=.18)
                ax.set_xlabel('Time point')
                ax.legend(fontsize=7, loc='best')
            axes[row, 0].set_ylabel('Value (dataset scale)\n' + ('Median' if row == 0 else 'Strongest')
                                    + ' uncapped background')
            axes[row, 1].text(.5, -.24, 'Target Q1-Q4: ' + ' / '.join(names[x] for x in target),
                              transform=axes[row, 1].transAxes, ha='center', fontsize=10)
            saved.append(dict(channel=channel, index=idx, background_id=bid, labels=list(target),
                              background_multipliers=(scales[:, bid]/old_scale[bid]).tolist(),
                              source_record=records[bid]))
        description = ('unscaled backgrounds; absolute column has shifted y-axis'
                       if args.raw_preview else 'less compressed weather backgrounds; same y-axis within each row')
        fig.suptitle(f'{channel}: {description}\n'
                     'Paired dataset construction preview, NOT model generations; fixed morphology/noise', fontsize=14)
        fig.subplots_adjust(top=.86, bottom=.12, hspace=.65, wspace=.12)
        fig.savefig(out / f'{channel}.png', dpi=150)
        plt.close(fig)
    modes = (['baseline_030', 'raw_centered', 'raw_absolute'] if args.raw_preview
             else [f'cap_{ratio}' for ratio in ratios])
    np.savez(out / 'paired_cases.npz', curves=np.stack(arrays), modes=modes,
             ratios=[] if args.raw_preview else ratios)
    (out / 'cases.json').write_text(json.dumps(dict(
        purpose='Dataset construction preview, not generated samples or updated captions',
        selection='Per channel: median and strongest uncapped maximum stage span in validation',
        amplitude=amplitude, noise_std=meta['noise_std'], modes=modes,
        ratios=None if args.raw_preview else ratios, cases=saved), indent=2))
    print(f'PASS: six paired cases; original curves reproduced and residuals unchanged. {out}', flush=True)


if __name__ == '__main__':
    main()