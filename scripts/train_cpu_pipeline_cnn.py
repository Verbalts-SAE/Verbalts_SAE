"""Train a CPU-specific CNN, selecting by validation whole-curve accuracy."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from contsg.eval.train_seg_classifier import train_epoch
from sae.train_morph_cnn import load_split
from sae.provenance import write_json, sha256_file
from scripts.cpu_pipeline_metrics import predict_cnn, accuracy_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train = load_split(args.data_root, 'train')
    labels = np.load(args.data_root / 'train_attrs_idx.npy').reshape(-1)
    weights = 1. / np.maximum(np.bincount(labels, minlength=4), 1)
    loader = DataLoader(train, batch_size=128,
        sampler=WeightedRandomSampler(weights[labels], len(labels), replacement=True))
    valid_x = np.load(args.data_root / 'valid_ts.npy')
    valid_y = np.load(args.data_root / 'valid_attrs_idx.npy')
    device = torch.device(args.device)
    model = PeakValleyClassifier1D(segment_len=43).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, stale, history = -1., 0, []
    path = args.output_dir / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        metrics = accuracy_report(predict_cnn(model, valid_x, device), valid_y)
        history.append(dict(epoch=epoch, train=loss, valid=metrics))
        if metrics['accr'] > best:
            best, stale = metrics['accr'], 0
            torch.save(model.state_dict(), path)
        else:
            stale += 1
        write_json(args.output_dir / 'report.json', dict(history=history,
            best_accr=best, selection_split='valid', test_evaluated=False,
            checkpoint_sha256=sha256_file(path)))
        print(epoch, metrics, flush=True)
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()