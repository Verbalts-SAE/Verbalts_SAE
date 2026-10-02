"""Structural/source audit; morphological quality still requires CNN validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_weather128 import (local_caption, shape_bank, prepare_source, COLUMNS,
                                        features, fit_thresholds, background_caption)
from sae.provenance import sha256_file, write_json


def audit(root):
    meta = json.loads((root / 'meta.json').read_text())
    source = prepare_source(pd.read_parquet(meta['source']))
    _, bank = shape_bank(meta['amplitude'])
    thresholds = fit_thresholds(features(np.load(root / 'train_backgrounds.npy')))
    assert thresholds == meta['caption_thresholds']
    occupied, report = set(), dict(status='PASS', quality_status='PENDING_CNN_AND_REVIEW', splits={})
    for split in ('train', 'valid', 'test'):
        x, y, ids, bg, caps = [np.load(root / f'{split}_{s}.npy') for s in
                               ('ts', 'attrs_idx', 'background_ids', 'backgrounds', 'text_caps')]
        records = json.loads((root / f'{split}_records.json').read_text())
        assert x.shape == (len(y), 128, 1) and y.shape == (len(x), 4)
        assert np.isfinite(x).all() and np.isin(y, [1, 2, 3]).all()
        combinations, counts = np.unique(y, axis=0, return_counts=True)
        assert len(combinations) == 81 and np.all(counts == len(bg))
        global_caps = [background_caption(f, thresholds) for f in features(bg)]
        for i, r in enumerate(records):
            lo, hi = r['source_start'], r['source_stop']
            assert hi-lo == 128 and not occupied.intersection(range(lo, hi))
            occupied.update(range(lo, hi))
            a, b = meta['splits'][split]['source_interval']
            assert a <= lo < hi <= b
            block = source.iloc[lo:hi]
            assert np.all(np.diff(block.timestamp.to_numpy()) == np.timedelta64(10, 'm'))
            assert block.original_row.tolist() == r['original_source_rows']
            raw = block[r['source_column']].to_numpy()
            np.testing.assert_allclose(bg[i], (raw-raw[0])*r['scale'])
            assert len(np.unique(y[ids == i], axis=0)) == 81
        for row, i, text in zip(y, ids, caps.reshape(-1)):
            assert text == f'{local_caption(row)} {global_caps[i]}'
        ratio = np.ptp(bg.reshape(-1, 4, 32), axis=2) / meta['amplitude']
        if meta.get('max_background_ratio') is not None:
            assert ratio.max() <= meta['max_background_ratio'] + 1e-8
        residual = x[:, :, 0] - bg[ids] - np.tile(bank, (len(bg), 1))
        report['splits'][split] = dict(samples=len(x), backgrounds=len(bg),
            background_ratio_quantiles=np.quantile(ratio, [0,.5,.9,.99,1]).tolist(),
            noise_std=float(residual.std()),
            hashes={s: sha256_file(root / f'{split}_{s}.npy') for s in ('ts','attrs_idx','text_caps')})
    write_json(root / 'audit.json', report)
    # Broader preview uses train only: each source channel and multiple backgrounds.
    from scripts.generate_weather_morph_preview import plt
    records = json.loads((root / 'train_records.json').read_text())
    x = np.load(root / 'train_ts.npy')[:, :, 0]
    bg = np.load(root / 'train_backgrounds.npy')
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for row, col in enumerate(COLUMNS):
        chosen = [i for i, r in enumerate(records) if r['source_column'] == col][:4]
        for ax, i, combo in zip(axes[row], chosen, (0, 13, 40, 80)):
            ax.plot(bg[i], color='gray', label='background')
            ax.plot(x[i*81+combo], label='sample')
            for boundary in (32,64,96):
                ax.axvline(boundary-.5, color='black', alpha=.2)
            ax.set_title(f'{col}; background={i}; combination={combo}')
            ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'preview_audit.png', dpi=140)
    plt.close(fig)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    audit(parser.parse_args().data_root)