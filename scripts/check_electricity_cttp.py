"""Check actual CTTP training inputs and write train/valid provenance only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from contsg.config.schema import ExperimentConfig
from contsg.config.model_validation import validate_model_config
from contsg.data.datamodule import TimeSeriesDataset


def check(config_path):
    cfg = validate_model_config(ExperimentConfig.from_yaml(config_path)).config
    data = cfg.data.data_folder
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_model_path, local_files_only=True)
    report = dict(seed=cfg.seed, model='cttp', selection='validation loss only',
                  test_used=False, text_encoding='online frozen LongCLIP; trainable projection', splits={})
    for split in ('train', 'valid'):
        dataset = TimeSeriesDataset(data, split=split, normalize=False)
        captions = np.load(data / f'{split}_text_caps.npy', allow_pickle=True).reshape(-1)
        assert len(dataset) == len(captions)
        assert all(dataset[i]['cap'] == str(c) for i, c in enumerate(captions)), 'Caption shadowing'
        assert all('global pattern' in str(c) for c in captions)
        lengths = [len(ids) for ids in tokenizer(captions.tolist(), truncation=False)['input_ids']]
        limit = min(tokenizer.model_max_length, 248)
        assert max(lengths) <= limit, 'Captions would be truncated'
        ts = np.load(data / f'{split}_ts.npy', mmap_mode='r')
        assert ts.shape == (len(captions), 128, 1) and np.isfinite(ts).all()
        report['splits'][split] = dict(samples=len(captions), max_tokens=max(lengths),
            hashes={name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                    for name in (f'{split}_ts.npy', f'{split}_text_caps.npy')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.config)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)