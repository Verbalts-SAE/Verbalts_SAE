"""Train a weather36 evaluation CNN; never load test or generated curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from contsg.eval.train_seg_classifier import train_epoch
from sae.provenance import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_split(root, split):
    if split not in ("train", "valid"):
        raise ValueError("Only train/valid allowed")
    curves = np.load(root / f"{split}_ts.npy")
    labels = np.load(root / f"{split}_attrs_idx.npy")
    if curves.shape != (len(curves), 36, 1) or not np.isfinite(curves).all():
        raise ValueError("Expected finite (N,36,1) curves")
    if labels.shape != (len(curves), 3) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer (N,3) labels")
    if not np.isin(labels, [1, 2, 3]).all():
        raise ValueError("Combined weather36 labels must be 1=single, 2=double, 3=sag")
    segments = torch.from_numpy(curves.reshape(-1, 1, 12).astype(np.float32))
    ids = labels.reshape(-1)
    peaks = torch.from_numpy(np.array([0, 1, 2, 0], dtype=np.int64)[ids])
    valleys = torch.from_numpy(np.array([0, 0, 0, 1], dtype=np.int64)[ids])
    return TensorDataset(segments, peaks, valleys), labels


def accuracy_report(predicted, labels):
    if predicted.shape != labels.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("Expected matching (N,3) labels")
    correct = predicted == labels
    confusion = np.zeros((4, 5), dtype=int)
    # Prediction 4 is an invalid peak/valley combination; do not force correctness.
    np.add.at(confusion, (labels.reshape(-1), predicted.reshape(-1)), 1)
    combos = {}
    for combo in np.unique(labels, axis=0):
        mask = (labels == combo).all(axis=1)
        combos["-".join(map(str, combo))] = {
            "n": int(mask.sum()), "accr": float(correct[mask].all(axis=1).mean())}
    return dict(accr=float(correct.all(axis=1).mean()),
                stage_accuracy=correct.mean(axis=0).tolist(),
                segment_accuracy=float(correct.mean()), confusion=confusion.tolist(),
                combinations=combos, n=len(labels))


@torch.no_grad()
def evaluate_curves(model, dataset, labels, device, batch_size):
    model.eval()
    predictions = []
    for x, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        peak, valley = model(x.to(device))
        peak, valley = peak.argmax(-1), valley.argmax(-1)
        ids = torch.full_like(peak, 4)
        ids[(peak == 0) & (valley == 0)] = 0
        ids[(peak == 1) & (valley == 0)] = 1
        ids[(peak == 2) & (valley == 0)] = 2
        ids[(peak == 0) & (valley == 1)] = 3
        predictions.append(ids.cpu().numpy())
    return accuracy_report(np.concatenate(predictions).reshape(-1, 3), labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/weather_three_stage_36_morph")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/weather36/cnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs/patience/batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train, _ = load_split(args.data_root, "train")
    valid, labels = load_split(args.data_root, "valid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "segment_cnn.pth"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {checkpoint}")
    model = PeakValleyClassifier1D(segment_len=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    best, stale, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, loader, optimizer, device)
        metrics = evaluate_curves(model, valid, labels, device, args.batch_size)
        history.append(dict(epoch=epoch, train=training, valid=metrics))
        print(f"epoch={epoch} valid_ACCR={metrics['accr']:.6f} segment_acc={metrics['segment_accuracy']:.6f}", flush=True)
        if metrics['accr'] > best:
            best, stale, best_epoch = metrics['accr'], 0, epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        write_json(args.output_dir / "progress.json", dict(best_epoch=best_epoch, history=history))
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    write_json(args.output_dir / "report.json", dict(
        best_epoch=best_epoch, valid=evaluate_curves(model, valid, labels, device, args.batch_size),
        checkpoint_sha256=sha256_file(checkpoint), seed=args.seed, test_evaluated=False,
        stage_bounds=[[0, 12], [12, 24], [24, 36]],
        labels={0: "nothing", 1: "single_peak", 2: "double_peaks", 3: "sag", 4: "invalid_prediction"},
        data_sha256={f"{s}_{k}": sha256_file(args.data_root / f"{s}_{k}.npy")
                     for s in ("train", "valid") for k in ("ts", "attrs_idx")},
        note="Validation used for classifier selection; not an independent classifier holdout. CNN is not a steering loss."))


if __name__ == "__main__":
    main()