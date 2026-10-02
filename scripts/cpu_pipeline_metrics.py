"""Shared CPU morphology metrics; invalid CNN head combinations count as errors."""
import numpy as np
import torch
from sae.shapes import split_segments


def accuracy_report(predicted, labels):
    predicted, labels = np.asarray(predicted), np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 3 or predicted.shape != labels.shape or not len(labels):
        raise ValueError('Expected nonempty matching (N,3) labels')
    if not np.isin(labels, range(4)).all() or not np.isin(predicted, range(5)).all():
        raise ValueError('Invalid class ids')
    matches = predicted == labels
    confusion = np.zeros((4, 5), dtype=int)
    np.add.at(confusion, (labels.ravel(), predicted.ravel()), 1)
    tp = np.diag(confusion[:, :4])
    recall = tp / np.maximum(confusion.sum(1), 1)
    precision = tp / np.maximum(confusion.sum(0)[:4], 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    nonempty = labels != 0
    nothing = ~nonempty.any(1)
    return dict(n=len(labels), accr=float(matches.all(1).mean()),
        segment_accuracy=float(matches.mean()), stage_accuracy=matches.mean(0).tolist(),
        macro_f1=float(f1.mean()), recall=recall.tolist(), precision=precision.tolist(),
        confusion=confusion.tolist(),
        event_accuracy=float(matches[nonempty].mean()) if nonempty.any() else None,
        nothing_curve_accuracy=float(matches[nothing].all(1).mean()) if nothing.any() else None)


@torch.no_grad()
def predict_cnn(model, curves, device, batch_size=256):
    model.eval()
    segments = split_segments(curves).reshape(-1, 1, 43).astype(np.float32)
    result = []
    for start in range(0, len(segments), batch_size):
        peaks, valleys = model(torch.from_numpy(segments[start:start + batch_size]).to(device))
        peak, valley = peaks.argmax(-1), valleys.argmax(-1)
        ids = torch.full_like(peak, 4)
        for index, (p, v) in enumerate(((0, 0), (1, 0), (2, 0), (0, 1))):
            ids[(peak == p) & (valley == v)] = index
        result.append(ids.cpu().numpy())
    return np.concatenate(result).reshape(-1, 3)