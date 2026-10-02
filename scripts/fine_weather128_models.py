"""Fine-grained 5-class CNN model and utilities for weather128 fine labels.

Labels per stage:
  1 = narrow single peak, 2 = wide single peak,
  3 = narrow double peaks, 4 = wide double peaks, 5 = sag
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

FINE_LABELS = (1, 2, 3, 4, 5)


def accuracy_report(predicted, labels):
    if labels.ndim != 2 or labels.shape[1] != 4 or predicted.shape != labels.shape or not len(labels):
        raise ValueError('Expected nonempty matching (N,4) arrays')
    correct = predicted == labels
    return dict(accr=float(correct.all(1).mean()), stage_accuracy=correct.mean(0).tolist(),
                segment_accuracy=float(correct.mean()), n=len(labels),
                correct=correct.tolist())


class FinePeakValleyClassifier1D(nn.Module):
    """Peak head extended to 5 classes (narrow/wide single/double + none)."""

    def __init__(self, segment_len: int):
        super().__init__()
        self.segment_len = segment_len
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
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc_peak = nn.Linear(128, 5)    # 0=none,1=narrow_single,2=wide_single,3=narrow_double,4=wide_double
        self.fc_valley = nn.Linear(128, 2)  # 0=none,1=sag

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)
        return self.fc_peak(x), self.fc_valley(x)


def load_fine_split(root, split):
    if split not in ('train', 'valid'):
        raise ValueError('Training must never load test')
    x = np.load(root / f'{split}_ts.npy')
    y = np.load(root / f'{split}_attrs_idx.npy')
    if x.shape != (len(y), 128, 1) or y.shape != (len(y), 4):
        raise ValueError('Expected (N,128,1)/(N,4)')
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, FINE_LABELS).all():
        raise ValueError('Invalid curves/labels')
    ids = y.reshape(-1)
    peak = np.array([0, 1, 2, 3, 4, 0])[ids]
    valley = np.array([0, 0, 0, 0, 0, 1])[ids]
    return TensorDataset(torch.tensor(x.reshape(-1, 1, 32), dtype=torch.float32),
                         torch.tensor(peak), torch.tensor(valley)), y


@torch.no_grad()
def evaluate_fine_curves(model, dataset, labels, device, batch_size=256):
    model.eval()
    parts = []
    for x, _, _ in DataLoader(dataset, batch_size=batch_size):
        peak, valley = [a.argmax(-1) for a in model(x.to(device))]
        ids = torch.full_like(peak, 0)
        for i, (p, v) in enumerate(((1, 0), (2, 0), (3, 0), (4, 0), (0, 1))):
            ids[(peak == p) & (valley == v)] = i + 1
        parts.append(ids.cpu().numpy())
    return accuracy_report(np.concatenate(parts).reshape(-1, 4), labels)
