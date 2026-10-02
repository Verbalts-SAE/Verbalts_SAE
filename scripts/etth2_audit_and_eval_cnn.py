"""Audit the shipped external CNN and evaluate our retrained segment CNNs on test.

Part 1 audits the dataset-shipped external weight ``etth2_2class_best.pt``
(claimed 97.2% test accuracy): rebuild its 2-class head architecture, load the
weights, and score it on the FULL test split.  This is a verification of the
external checkpoint, not an adoption of it.

Part 2 evaluates our own retrained ``PeakValleyClassifier1D`` checkpoints
(one per seed) on the FULL test split, reporting segment accuracy and the
whole-curve exact match (ACCR: all three segments correct).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.provenance import sha256_file, write_json
from sae.shapes import split_segments


class Etth2TwoClassCNN(nn.Module):
    """Backbone shared with PeakValleyClassifier1D + a single 2-class head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc_shared = nn.Linear(256, 128)
        self.fc_head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.mean(dim=2)
        x = torch.relu(self.fc_shared(x))
        return self.fc_head(x)


def evaluate_external(root: Path, device: torch.device, batch_size: int = 512) -> dict:
    checkpoint = root / "etth2_2class_best.pt"
    model = Etth2TwoClassCNN().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    labels = np.load(root / "test_labels.npy")  # (N, 3) binary
    curves = np.load(root / "test_ts.npy")[..., 0]
    segments = split_segments(curves)  # (N, 3, 43)
    predictions = []
    for start in range(0, len(segments), batch_size):
        chunk = torch.from_numpy(
            segments[start:start + batch_size].reshape(-1, 1, 43).astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits = model(chunk)
        predictions.append(logits.argmax(-1).cpu().numpy())
    predicted = np.concatenate(predictions).reshape(len(segments), 3)
    correct = predicted == labels
    accr = float(correct.all(axis=1).mean())
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_test": int(len(labels)),
        "segment_accuracy": float(correct.mean()),
        "per_stage_accuracy": {
            stage: float(correct[:, index].mean())
            for index, stage in enumerate(("beginning", "middle", "end"))
        },
        "whole_curve_exact_match_accr": accr,
        "claimed_test_accuracy": 0.972,
        "per_class_accuracy": {
            f"segment_label_{label}": float(
                correct[labels == label].mean()
            ) if (labels == label).any() else float("nan")
            for label in (0, 1)
        },
    }
    return report


def evaluate_retrained(
    data_root: Path, checkpoint: Path, device: torch.device, batch_size: int = 512
) -> dict:
    model = PeakValleyClassifier1D(segment_len=43).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    labels = np.load(data_root / "test_attrs_idx.npy")  # (N, 3) in {1, 2}
    curves = np.load(data_root / "test_ts.npy")[..., 0]
    segments = split_segments(curves)  # (N, 3, 43)
    peak_targets = labels.reshape(-1).astype(np.int64)
    valley_targets = np.zeros_like(peak_targets)
    peak_pred, valley_pred = [], []
    for start in range(0, len(segments), batch_size):
        chunk = torch.from_numpy(
            segments[start:start + batch_size].reshape(-1, 1, 43).astype(np.float32)
        ).to(device)
        with torch.no_grad():
            peak_logits, valley_logits = model(chunk)
        peak_pred.append(peak_logits.argmax(-1).cpu().numpy())
        valley_pred.append(valley_logits.argmax(-1).cpu().numpy())
    peak_pred = np.concatenate(peak_pred)
    valley_pred = np.concatenate(valley_pred)
    peak_correct = (peak_pred == peak_targets).reshape(len(segments), 3)
    valley_correct = (valley_pred == valley_targets).reshape(len(segments), 3)
    joint = peak_correct & valley_correct
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_test": int(len(labels)),
        "peak_accuracy": float(peak_correct.mean()),
        "valley_accuracy": float(valley_correct.mean()),
        "segment_joint_accuracy": float(joint.mean()),
        "whole_curve_exact_match_accr": float(joint.all(axis=1).mean()),
        "per_stage_joint_accuracy": {
            stage: float(joint[:, index].mean())
            for index, stage in enumerate(("beginning", "middle", "end"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {"external_checkpoint_audit": evaluate_external(args.data_root, device)}
    report["retrained_cnns"] = {}
    for seed_dir in args.seed_dirs:
        checkpoint = seed_dir / "cnn" / "segment_cnn.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing retrained CNN: {checkpoint}")
        report["retrained_cnns"][seed_dir.name] = evaluate_retrained(
            args.data_root, checkpoint, device
        )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
