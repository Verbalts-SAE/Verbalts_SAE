import numpy as np
import pytest
import torch

from scripts.prepare_weather128 import backgrounds
from scripts.train_weather128_cnn import accuracy_report, load_split
from sae.shape_classifier import SAEClassClassifier, classifier_metrics


def test_background_cap_preserves_continuity():
    n = 128 * 12
    values = np.column_stack([np.arange(n), np.arange(n)+1, np.arange(n)+1000])
    times = np.datetime64('2020-01-01') + np.arange(n)*np.timedelta64(10, 'm')
    bg, records = backgrounds(values, times, (0,n), 6, 42, max_stage_span=.36)
    assert np.ptp(bg.reshape(-1,4,32),axis=2).max() <= .36+1e-8
    assert np.all(bg[:,32] > 0)
    assert all(r['scale'] > 0 for r in records)


def test_joint_accuracy_and_exact_quarters(tmp_path):
    labels = np.array([[1,2,3,1],[3,2,1,3]])
    predicted = labels.copy()
    predicted[0,3] = 4
    assert accuracy_report(predicted, labels)['accr'] == .5
    with pytest.raises(ValueError):
        accuracy_report(labels[:,:3], labels[:,:3])
    np.save(tmp_path/'train_ts.npy', np.arange(256).reshape(2,128,1).astype('float32'))
    np.save(tmp_path/'train_attrs_idx.npy', labels)
    dataset, _ = load_split(tmp_path, 'train')
    assert dataset.tensors[0][:,0,0].tolist() == list(range(0,256,32))
    with pytest.raises(ValueError):
        load_split(tmp_path, 'test')


def test_four_head_mlp_gradient_and_metrics():
    model = SAEClassClassifier(8, 5, hidden_dim=16, num_heads=4, dropout=0)
    x = torch.randn(3,5,8, requires_grad=True)
    logits = model(x)
    assert logits.shape == (3,4,4)
    logits[:,:,1].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    report = classifier_metrics(model, x.detach(), logits.detach().argmax(-1), torch.device('cpu'), 2)
    assert report['accr'] == 1 and report['num_heads'] == 4
    assert list(report['stage_accuracy']) == ['first','second','third','fourth']