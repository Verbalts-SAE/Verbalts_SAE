"""Build continuous single-channel weather128 with four labelled quarters."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contsg.data.datasets.semisynth_morph import InjectionConfig, morphology_residual
from scripts.generate_weather_morph_dataset import SOURCE, COLUMNS, CHANNELS, prepare_source
from scripts.prepare_weather36_captions import features, fit_thresholds, background_caption, embed

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('single_peak', 'double_peaks', 'sag')
STAGES = ('first', 'second', 'third', 'fourth')


def intervals(size):
    a, b = int(size * .8), int(size * .9)
    return {'train': (0, a), 'valid': (a + 128, b), 'test': (b + 128, size)}


def backgrounds(values, times, interval, count, seed, max_stage_span=None):
    if max_stage_span is not None and (not np.isfinite(max_stage_span) or max_stage_span <= 0):
        raise ValueError('max_stage_span must be finite and positive')
    lo, hi = interval
    if count < 1 or not 0 <= lo < hi <= len(values):
        raise ValueError('Invalid count or interval')
    candidates = []
    for start in range(lo, hi - 127, 128):
        block = values[start:start+128]
        if (np.isfinite(block).all() and (block[:, 1] >= 0).all()
                and (block[:, 2] > 0).all()
                and np.all(np.diff(times[start:start+128]) == np.timedelta64(10, 'm'))):
            candidates.append(start)
    if len(candidates) < count:
        raise ValueError(f'Need {count} disjoint blocks; have {len(candidates)}')
    rng = np.random.default_rng(seed)
    starts = rng.choice(candidates, count, replace=False)
    channels = rng.permutation(np.arange(count) % 3)
    arrays, records = [], []
    for i, (start, channel) in enumerate(zip(starts, channels)):
        start, channel = int(start), int(channel)
        raw = values[start:start+128, channel].astype(float)
        span = np.ptp(raw)
        step = np.abs(np.diff(raw))
        gentle = not span or (step.max()/span <= .25 and step.sum()/span <= 2.5)
        scale = 1. if channel == 2 else (.4 if gentle else .2)
        if max_stage_span is not None:
            stage_span = np.ptp(raw.reshape(4, 32), axis=1).max()
            if stage_span > 0:
                scale = min(scale, max_stage_span / stage_span)
        arrays.append((raw - raw[0]) * scale)
        records.append(dict(background_id=i, source_start=start, source_stop=start+128,
                            channel=CHANNELS[channel], source_column=COLUMNS[channel],
                            scale=scale, raw_first=float(raw[0]),
                            time_start=str(times[start]), time_end=str(times[start+127])))
    return np.stack(arrays), records


def shape_bank(amplitude):
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Amplitude must be finite and positive')
    labels = np.array(list(product((1, 2, 3), repeat=4)), dtype=np.int64)
    residuals = np.stack([np.concatenate([morphology_residual(
        NAMES[i-1], 32, InjectionConfig(amplitude=amplitude)) for i in row]) for row in labels])
    return labels, residuals


def local_caption(labels):
    phrases = ('a single peak', 'double peaks', 'a sag')
    if len(labels) != 4 or not np.isin(labels, [1, 2, 3]).all():
        raise ValueError('Expected four morphology labels in [1,3]')
    return ' '.join(f'The {stage} quarter has {phrases[int(label)-1]}.'
                    for stage, label in zip(STAGES, labels))


def prepare(source, output, counts=(400, 50, 50), seed=20260928, amplitude=1.2, noise=.08,
            max_background_ratio=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if len(counts) != 3 or min(counts) < 3 or not np.isfinite(noise) or noise < 0:
        raise ValueError('Invalid background counts or noise')
    labels, residuals = shape_bank(amplitude)
    frame = prepare_source(pd.read_parquet(source, columns=['Date Time', *COLUMNS]))
    values, times = frame[list(COLUMNS)].to_numpy(), frame.timestamp.to_numpy()
    splits = intervals(len(frame))
    prepared = {}
    for index, (split, bounds) in enumerate(splits.items()):
        bg, records = backgrounds(values, times, bounds, counts[index], seed+index,
                                  None if max_background_ratio is None else amplitude * max_background_ratio)
        for r in records:
            r['original_source_rows'] = frame.original_row.iloc[r['source_start']:r['source_stop']].tolist()
        prepared[split] = bg, records, features(bg)
    thresholds = fit_thresholds(prepared['train'][2])
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(source=str(source), seq_length=128, sequence_length=128, n_var=1,
                stage_bounds=[[0,32],[32,64],[64,96],[96,128]], stage_names=STAGES,
                label_names=['nothing', *NAMES], normalize=False, seed=seed,
                amplitude=amplitude, noise_std=noise, caption_thresholds=thresholds,
                max_background_ratio=max_background_ratio,
                construction='One channel per sample, 128 consecutive rows; no stitching or resampling',
                split_policy='Chronological 80/10/10, 128-row embargo, disjoint background blocks',
                background_transform='(raw - raw[0]) * scale once per entire sample',
                splits={})
    for index, (split, (bg, records, matrix)) in enumerate(prepared.items()):
        ids = np.repeat(np.arange(len(bg)), len(labels))
        targets = np.tile(labels, (len(bg), 1))
        rng = np.random.default_rng(seed+100+index)
        curves = bg[ids] + np.tile(residuals, (len(bg), 1))
        curves += rng.normal(0, noise, curves.shape)
        local = np.array([local_caption(row) for row in targets])
        global_caps = [background_caption(row, thresholds) for row in matrix]
        caps = np.array([f'{text} {global_caps[i]}' for text, i in zip(local, ids)])
        assert curves.shape == (len(bg)*81, 128) and np.isfinite(curves).all()
        for suffix, array in dict(ts=curves.astype(np.float32)[..., None], attrs_idx=targets,
                                  text_caps=caps[:, None], text_caps_local=local[:, None],
                                  background_ids=ids, backgrounds=bg, background_features=matrix).items():
            np.save(output / f'{split}_{suffix}.npy', array)
        (output / f'{split}_records.json').write_text(json.dumps(records, indent=2))
        meta['splits'][split] = dict(source_interval=splits[split], backgrounds=len(bg), samples=len(ids))
    (output / 'meta.json').write_text(json.dumps(meta, indent=2))
    from scripts.generate_weather_morph_preview import plt
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    train = np.load(output / 'train_ts.npy')[:, :, 0]
    for ax, i in zip(axes.flat, (0, 13, 26, 30, 40, 50, 54, 67, 80)):
        ax.plot(prepared['train'][0][0], color='gray', alpha=.6, label='background')
        ax.plot(train[i], label='sample')
        for boundary in (32, 64, 96):
            ax.axvline(boundary-.5, color='black', alpha=.3)
        ax.set_title('/'.join(NAMES[k-1] for k in labels[i]), fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'preview.png', dpi=140)
    plt.close(fig)
    print(json.dumps(meta['splits']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/weather_four_stage_128_morph')
    parser.add_argument('--embed-only', action='store_true')
    parser.add_argument('--max-background-ratio', type=float)
    parser.add_argument('--amplitude', type=float, default=1.2)
    args = parser.parse_args()
    if args.embed_only:
        embed(args.output, 'cuda:0')
    else:
        prepare(args.source, args.output, max_background_ratio=args.max_background_ratio,
                amplitude=args.amplitude)