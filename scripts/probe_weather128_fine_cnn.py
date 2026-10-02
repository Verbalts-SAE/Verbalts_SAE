"""Probe CNN reliability for fine-grained 5-class morphology labels.

Five classes per stage:
  1 = narrow single peak (width .05)
  2 = wide single peak   (width .14)
  3 = narrow double peaks
  4 = wide double peaks
  5 = sag

Builds a small probe dataset with the same background construction as the
weather128 pipeline (max_background_ratio), trains a 5-class CNN, and reports
its real-data joint ACCR. The CNN must stay >=99% for the fine labels to be
usable as the difficulty axis.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from contsg.data.datasets.semisynth_morph import InjectionConfig, morphology_residual
from contsg.eval.train_seg_classifier import train_epoch
from sae.provenance import write_json

ROOT = Path(__file__).resolve().parents[1]

FINE_NAMES = ('narrow_single_peak', 'wide_single_peak', 'narrow_double_peaks',
              'wide_double_peaks', 'sag')
MORPH_KEYS = ('single_peak', 'single_peak', 'double_peaks', 'double_peaks', 'sag')
WIDTHS = (0.05, 0.14, 0.05, 0.14, 0.08)


def fine_shape_bank(amplitude):
    from itertools import product
    labels = np.array(list(product(range(1, 6), repeat=4)), dtype=np.int64)
    residuals = []
    for row in labels:
        parts = [morphology_residual(MORPH_KEYS[i - 1], 32,
                                     InjectionConfig(amplitude=amplitude, width=WIDTHS[i - 1]))
                 for i in row]
        residuals.append(np.concatenate(parts))
    return labels, np.stack(residuals)


class FinePeakValleyClassifier1D(torch.nn.Module):
    """Peak head extended to 5 classes (narrow/wide single/double + none)."""

    def __init__(self, segment_len: int):
        super().__init__()
        self.segment_len = segment_len
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.pool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.pool2 = torch.nn.MaxPool1d(2)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = torch.nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc_shared = torch.nn.Linear(256, 128)
        self.dropout = torch.nn.Dropout(0.4)
        self.fc_peak = torch.nn.Linear(128, 5)    # 0=none,1=narrow_single,2=wide_single,3=narrow_double,4=wide_double
        self.fc_valley = torch.nn.Linear(128, 2)  # 0=none,1=sag

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        x = torch.nn.functional.relu(self.fc_shared(x))
        x = self.dropout(x)
        return self.fc_peak(x), self.fc_valley(x)


def accuracy_report(predicted, labels):
    correct = predicted == labels
    return dict(accr=float(correct.all(1).mean()), stage_accuracy=correct.mean(0).tolist(),
                segment_accuracy=float(correct.mean()), n=len(labels))


@torch.no_grad()
def evaluate_fine(model, dataset, labels, device, batch_size=256):
    model.eval()
    parts = []
    for x, _, _ in DataLoader(dataset, batch_size=batch_size):
        peak, valley = [a.argmax(-1) for a in model(x.to(device))]
        ids = torch.full_like(peak, 0)
        for i, (p, v) in enumerate(((1, 0), (2, 0), (3, 0), (4, 0), (0, 1))):
            ids[(peak == p) & (valley == v)] = i + 1
        parts.append(ids.cpu().numpy())
    return accuracy_report(np.concatenate(parts).reshape(-1, 4), labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--counts', type=int, nargs=3, default=(200, 50, 50))
    p.add_argument('--amplitude', type=float, default=1.2)
    p.add_argument('--max-background-ratio', type=float, default=0.6)
    p.add_argument('--noise', type=float, default=0.08)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    from scripts.prepare_weather128 import backgrounds, intervals, prepare_source
    from scripts.generate_weather_morph_dataset import SOURCE, COLUMNS, CHANNELS
    import pandas as pd
    frame = prepare_source(pd.read_parquet(SOURCE, columns=['Date Time', *COLUMNS]))
    values, times = frame[list(COLUMNS)].to_numpy(), frame.timestamp.to_numpy()
    splits = intervals(len(frame))
    labels, residuals = fine_shape_bank(args.amplitude)
    prepared = {}
    for index, (split, bounds) in enumerate(splits.items()):
        bg, _ = backgrounds(values, times, bounds, args.counts[index], args.seed + index,
                            args.amplitude * args.max_background_ratio)
        prepared[split] = bg
    report = dict(amplitude=args.amplitude, max_background_ratio=args.max_background_ratio,
                  noise=args.noise, counts=list(args.counts), results=[])
    model = FinePeakValleyClassifier1D(segment_len=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    datasets = {}
    for split in ('train', 'valid'):
        bg = prepared[split]
        ids = np.repeat(np.arange(len(bg)), len(labels))
        rng = np.random.default_rng(args.seed + 100 + ('train', 'valid').index(split))
        curves = bg[ids] + np.tile(residuals, (len(bg), 1))
        curves += rng.normal(0, args.noise, curves.shape)
        y = np.tile(labels, (len(bg), 1))
        flat = y.reshape(-1)
        peak = np.array([0, 1, 2, 3, 4, 0])[flat]
        valley = np.array([0, 0, 0, 0, 0, 1])[flat]
        datasets[split] = (TensorDataset(torch.tensor(curves.reshape(-1, 1, 32), dtype=torch.float32),
                                         torch.tensor(peak), torch.tensor(valley)), y)
    loader = DataLoader(datasets['train'][0], batch_size=256, shuffle=True)
    best, stale = -1., 0
    for epoch in range(1, 101):
        loss = train_epoch(model, loader, optimizer, device)
        metrics = evaluate_fine(model, datasets['valid'][0], datasets['valid'][1], device)
        metrics.pop('stage_accuracy')
        print(epoch, metrics, flush=True)
        if metrics['accr'] > best:
            best, stale = metrics['accr'], 0
        else:
            stale += 1
        if stale >= 15:
            break
    report['results'].append(dict(num_classes=5, cnn_real_valid_accr=best))
    report['cnn_real_valid_accr'] = best
    write_json(out / 'probe_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
