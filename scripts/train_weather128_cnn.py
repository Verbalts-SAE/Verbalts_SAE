"""Four-quarter evaluation CNN, selected on validation joint ACCR only."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from contsg.eval.train_seg_classifier import train_epoch
from sae.provenance import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/weather_four_stage_128_morph_v2'


def accuracy_report(predicted, labels):
    if labels.ndim != 2 or labels.shape[1] != 4 or predicted.shape != labels.shape or not len(labels):
        raise ValueError('Expected nonempty matching (N,4) arrays')
    correct = predicted == labels
    return dict(accr=float(correct.all(1).mean()), stage_accuracy=correct.mean(0).tolist(),
                segment_accuracy=float(correct.mean()), n=len(labels),
                correct=correct.tolist())


def load_split(root, split):
    if split not in ('train', 'valid'):
        raise ValueError('Training must never load test')
    x = np.load(root / f'{split}_ts.npy')
    y = np.load(root / f'{split}_attrs_idx.npy')
    if x.shape != (len(y), 128, 1) or y.shape != (len(y), 4):
        raise ValueError('Expected (N,128,1)/(N,4)')
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [1, 2, 3]).all():
        raise ValueError('Invalid curves/labels')
    ids = y.reshape(-1)
    return TensorDataset(torch.tensor(x.reshape(-1, 1, 32), dtype=torch.float32),
        torch.tensor(np.array([0, 1, 2, 0])[ids]),
        torch.tensor(np.array([0, 0, 0, 1])[ids])), y


@torch.no_grad()
def evaluate_curves(model, dataset, labels, device, batch_size=256):
    model.eval()
    parts = []
    for x, _, _ in DataLoader(dataset, batch_size=batch_size):
        peak, valley = [a.argmax(-1) for a in model(x.to(device))]
        ids = torch.full_like(peak, 4)
        for i, (p, v) in enumerate(((0, 0), (1, 0), (2, 0), (0, 1))):
            ids[(peak == p) & (valley == v)] = i
        parts.append(ids.cpu().numpy())
    return accuracy_report(np.concatenate(parts).reshape(-1, 4), labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train, _ = load_split(args.data_root, 'train')
    valid, labels = load_split(args.data_root, 'valid')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = PeakValleyClassifier1D(segment_len=32).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=256, shuffle=True)
    best, stale, history = -1., 0, []
    checkpoint = args.output_dir / 'segment_cnn.pth'
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, torch.device(args.device))
        metrics = evaluate_curves(model, valid, labels, args.device)
        metrics.pop('correct')
        history.append(dict(epoch=epoch, train=loss, valid=metrics))
        if metrics['accr'] > best:
            best, stale = metrics['accr'], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        write_json(args.output_dir / 'report.json', dict(history=history, best_accr=best,
            stage_bounds=[[0,32],[32,64],[64,96],[96,128]], seed=args.seed,
            selection_split='valid', test_evaluated=False,
            checkpoint_sha256=sha256_file(checkpoint)))
        print(epoch, metrics, flush=True)
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()