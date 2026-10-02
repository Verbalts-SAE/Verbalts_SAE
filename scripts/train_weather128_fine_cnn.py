"""Fine 5-class four-quarter evaluation CNN, selected on validation joint ACCR."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from contsg.eval.train_seg_classifier import train_epoch
from sae.provenance import sha256_file, write_json
from scripts.fine_weather128_models import FinePeakValleyClassifier1D, evaluate_fine_curves, load_fine_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/weather_four_stage_128_morph_fine'


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
    train, _ = load_fine_split(args.data_root, 'train')
    valid, labels = load_fine_split(args.data_root, 'valid')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = FinePeakValleyClassifier1D(segment_len=32).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=256, shuffle=True)
    best, stale, history = -1., 0, []
    checkpoint = args.output_dir / 'segment_cnn.pth'
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, torch.device(args.device))
        metrics = evaluate_fine_curves(model, valid, labels, args.device)
        metrics.pop('correct')
        history.append(dict(epoch=epoch, train=loss, valid=metrics))
        if metrics['accr'] > best:
            best, stale = metrics['accr'], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        write_json(args.output_dir / 'report.json', dict(history=history, best_accr=best,
            stage_bounds=[[0, 32], [32, 64], [64, 96], [96, 128]], seed=args.seed,
            num_classes_per_stage=5, selection_split='valid', test_evaluated=False,
            checkpoint_sha256=sha256_file(checkpoint)))
        print(epoch, metrics, flush=True)
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()
