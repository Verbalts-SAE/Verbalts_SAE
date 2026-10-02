"""Train the existing segment CNN on predefined morphology train/valid splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from contsg.eval.train_seg_classifier import evaluate, train_epoch
from sae.provenance import sha256_file, write_json
from sae.shapes import split_segments


def load_split(root: Path, split: str) -> TensorDataset:
    curves = np.load(root / f"{split}_ts.npy")
    labels = np.load(root / f"{split}_attrs_idx.npy")
    if labels.shape != (len(curves), 3):
        raise ValueError("expected three morphology labels per curve")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("morphology labels must be integers in [0, 3]")
    segments = split_segments(curves).reshape(-1, 1, 43).astype(np.float32)
    if not np.isfinite(segments).all():
        raise ValueError("non-finite curve values")
    ids = labels.reshape(-1)
    peaks = np.asarray([0, 1, 2, 0], dtype=np.int64)[ids]
    valleys = np.asarray([0, 0, 0, 1], dtype=np.int64)[ids]
    return TensorDataset(torch.from_numpy(segments), torch.from_numpy(peaks), torch.from_numpy(valleys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train = load_split(args.data_root, "train")
    valid = load_split(args.data_root, "valid")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)
    model = PeakValleyClassifier1D(segment_len=43).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "segment_cnn.pth"
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        valid_metrics = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(f"epoch={epoch} train={train_metrics} valid={valid_metrics}", flush=True)
        if valid_metrics["joint_acc"] > best:
            best, stale, best_epoch = valid_metrics["joint_acc"], 0, epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    write_json(args.output_dir / "segment_cnn_report.json", {
        "best_epoch": best_epoch, "valid": evaluate(model, valid_loader, device),
        "history": history, "test_evaluated": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "data_sha256": {f"{s}_{kind}": sha256_file(args.data_root / f"{s}_{kind}.npy")
                        for s in ("train", "valid") for kind in ("ts", "attrs_idx")},
        "seed": args.seed, "segment_slices": [[0, 43], [43, 86], [85, 128]],
    })


if __name__ == "__main__":
    main()