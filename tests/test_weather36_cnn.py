import numpy as np
import pytest
import torch

from scripts.train_weather36_cnn import load_split, accuracy_report
from contsg.eval.metrics.segment import PeakValleyClassifier1D


def test_segments_and_labels(tmp_path):
    np.save(tmp_path / 'train_ts.npy', np.arange(72).reshape(2, 36, 1))
    np.save(tmp_path / 'train_attrs_idx.npy', np.array([[1, 2, 3], [3, 2, 1]]))
    ds, labels = load_split(tmp_path, 'train')
    assert ds.tensors[0].shape == (6, 1, 12)
    assert ds.tensors[0][2, 0].tolist() == list(range(24, 36))
    assert ds.tensors[1].tolist() == [1, 2, 0, 0, 2, 1]
    assert ds.tensors[2].tolist() == [0, 0, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        load_split(tmp_path, 'test')


def test_accr_not_segment_accuracy():
    labels = np.array([[1, 2, 3], [1, 2, 3]])
    pred = np.array([[1, 2, 3], [1, 2, 4]])
    report = accuracy_report(pred, labels)
    assert report['accr'] == .5
    assert report['segment_accuracy'] == pytest.approx(5 / 6)
    assert report['confusion'][3][4] == 1


def test_cnn_length12_backward():
    model = PeakValleyClassifier1D(12)
    peaks, valleys = model(torch.randn(6, 1, 12))
    assert peaks.shape == (6, 3)
    assert valleys.shape == (6, 2)
    (peaks.sum() + valleys.sum()).backward()
    assert torch.isfinite(model.conv1.weight.grad).all()