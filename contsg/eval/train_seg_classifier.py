"""
Training pipeline for segment shape classifier (PeakValleyClassifier1D).

Trains a 1D-CNN with dual heads (peak_count 3-class, valley_count 2-class)
on per-segment data from classified_shape_label_from_caption/.

Usage:
    python -m contsg.eval.train_seg_classifier \
        --data-root data/classified_shape_label_from_caption \
        --save-dir segment_classifiers \
        --epochs 100 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from contsg.eval.metrics.segment import PeakValleyClassifier1D

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Shape → Label Mapping
# ==============================================================================

SHAPE_TO_PEAK = {"nothing": 0, "peak": 1, "double_peak": 2, "sag": 0}
SHAPE_TO_VALLEY = {"sag": 1, "nothing": 0, "peak": 0, "double_peak": 0}


def shape_to_labels(shape: str) -> Tuple[int, int]:
    """Map a segment shape string to (peak_count, valley_count)."""
    return SHAPE_TO_PEAK[shape], SHAPE_TO_VALLEY[shape]


# ==============================================================================
# Dataset
# ==============================================================================


class SegShapeDataset(Dataset):
    """
    Dataset for per-segment shape classification.

    Loads raw.pt from all class subdirectories under data_root,
    splits each 128-step time series into 3 segments, and derives
    peak_count / valley_count labels from segment_shapes.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.samples: List[Tuple[Tensor, int, int]] = []

        class_dirs = sorted(
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d / "raw.pt").exists()
        )

        for class_dir in class_dirs:
            raw = torch.load(class_dir / "raw.pt", map_location="cpu", weights_only=False)
            ts_list = raw["samples"]       # list of (128, 1) tensors
            shapes_list = raw["segment_shapes"]  # list of [shape1, shape2, shape3]

            for ts, shapes in zip(ts_list, shapes_list):
                ts = ts.squeeze(-1)  # (128,)
                # Split into 3 equal-length segments: 43 each (last overlaps 1 step)
                segs = [ts[:43], ts[43:86], ts[85:128]]
                for seg, shape in zip(segs, shapes):
                    peak, valley = shape_to_labels(shape)
                    self.samples.append((seg.clone().detach(), peak, valley))

        logger.info("Loaded %d segments from %d class directories", len(self.samples), len(class_dirs))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        ts, peak, valley = self.samples[idx]
        return (
            ts.unsqueeze(0).float(),           # (1, seg_len)
            torch.tensor(peak, dtype=torch.long),
            torch.tensor(valley, dtype=torch.long),
        )


# ==============================================================================
# Training Utilities
# ==============================================================================


@torch.no_grad()
def evaluate(model: PeakValleyClassifier1D, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model, returning accuracies and per-class precision/recall."""
    model.eval()
    all_peak_logits, all_valley_logits = [], []
    all_peak_t, all_valley_t = [], []

    for x, pt, vt in loader:
        x, pt, vt = x.to(device), pt.to(device), vt.to(device)
        pl, vl = model(x)
        all_peak_logits.append(pl.cpu())
        all_valley_logits.append(vl.cpu())
        all_peak_t.append(pt.cpu())
        all_valley_t.append(vt.cpu())

    pl = torch.cat(all_peak_logits)
    vl = torch.cat(all_valley_logits)
    pt = torch.cat(all_peak_t)
    vt = torch.cat(all_valley_t)

    peak_pred = pl.argmax(1)
    valley_pred = vl.argmax(1)
    joint = (peak_pred == pt) & (valley_pred == vt)

    metrics = {
        "peak_acc": (peak_pred == pt).float().mean().item(),
        "valley_acc": (valley_pred == vt).float().mean().item(),
        "joint_acc": joint.float().mean().item(),
        "n": len(pt),
    }
    # Per-class precision/recall for peak (3 classes)
    for c in range(3):
        tp = ((peak_pred == c) & (pt == c)).sum().item()
        fp = ((peak_pred == c) & (pt != c)).sum().item()
        fn = ((peak_pred != c) & (pt == c)).sum().item()
        metrics[f"peak_cls{c}_prec"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f"peak_cls{c}_rec"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return metrics


def train_epoch(
    model: PeakValleyClassifier1D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train one epoch, return average metrics."""
    model.train()
    tot_loss = tot_pa = tot_va = tot_ja = 0.0
    n = 0

    for x, pt, vt in loader:
        x, pt, vt = x.to(device), pt.to(device), vt.to(device)
        optimizer.zero_grad()
        pl, vl = model(x)
        loss = F.cross_entropy(pl, pt) + F.cross_entropy(vl, vt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pa = (pl.argmax(1) == pt).float().mean().item()
            va = (vl.argmax(1) == vt).float().mean().item()
            ja = ((pl.argmax(1) == pt) & (vl.argmax(1) == vt)).float().mean().item()

        tot_loss += loss.item()
        tot_pa += pa
        tot_va += va
        tot_ja += ja
        n += 1

    return {
        "loss": tot_loss / n,
        "peak_acc": tot_pa / n,
        "valley_acc": tot_va / n,
        "joint_acc": tot_ja / n,
    }


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup: int, total: int, min_factor: float = 0.01):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.min = min_factor
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        lr = self._lr()
        for g, b in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * (b / self.base_lrs[0])

    def _lr(self):
        if self.epoch <= self.warmup:
            return self.base_lrs[0] * self.epoch / max(1, self.warmup)
        p = (self.epoch - self.warmup) / max(1, self.total - self.warmup)
        return self.base_lrs[0] * (self.min + (1 - self.min) * 0.5 * (1 + math.cos(math.pi * p)))


# ==============================================================================
# Main Training Pipeline
# ==============================================================================


def train(
    data_root: str,
    save_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = "cuda",
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
) -> str:
    """
    Train PeakValleyClassifier1D and save best checkpoint.

    Returns absolute path to best checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    ds = SegShapeDataset(data_root)
    n = len(ds)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(dev.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=(dev.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    # --- Model ---
    model = PeakValleyClassifier1D(segment_len=43).to(dev)
    logger.info("Model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, warmup=5, total=epochs)

    # --- Training loop ---
    best_joint = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, dev)
        scheduler.step()
        val_m = evaluate(model, val_loader, dev)

        logger.info(
            "Epoch %3d | lr=%.6f | train loss=%.4f p=%.4f v=%.4f j=%.4f | val p=%.4f v=%.4f j=%.4f",
            epoch, optimizer.param_groups[0]["lr"],
            train_m["loss"], train_m["peak_acc"], train_m["valley_acc"], train_m["joint_acc"],
            val_m["peak_acc"], val_m["valley_acc"], val_m["joint_acc"],
        )

        if val_m["joint_acc"] > best_joint:
            best_joint = val_m["joint_acc"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), save_path / "peak_valley_classifier_best.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # --- Final test ---
    model.load_state_dict(torch.load(save_path / "peak_valley_classifier_best.pth", map_location=dev))
    test_m = evaluate(model, test_loader, dev)

    logger.info("=" * 60)
    logger.info("Test (best epoch=%d):", best_epoch)
    logger.info("  peak_acc=%.4f  valley_acc=%.4f  joint_acc=%.4f", test_m["peak_acc"], test_m["valley_acc"], test_m["joint_acc"])
    for c in range(3):
        logger.info("  peak class %d: prec=%.4f  rec=%.4f", c, test_m[f"peak_cls{c}_prec"], test_m[f"peak_cls{c}_rec"])
    logger.info("=" * 60)

    # --- Report ---
    report = {
        "model": "PeakValleyClassifier1D",
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "best_epoch": best_epoch, "epochs_done": epoch,
        "hyperparams": {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size},
        "test": test_m,
    }
    rp = save_path / "peak_valley_classifier_report.json"
    json.dump(report, open(rp, "w"), indent=2, default=float)
    logger.info("Report saved to %s", rp)

    return str((save_path / "peak_valley_classifier_best.pth").absolute())


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train PeakValleyClassifier1D")
    parser.add_argument("--data-root", required=True, help="Path to classified_shape_label_from_caption/")
    parser.add_argument("--save-dir", required=True, help="Directory to save checkpoint and report")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ckpt = train(
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    logger.info("Done. Best checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
