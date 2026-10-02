import numpy as np
import pytest
from scripts.cpu_pipeline_metrics import accuracy_report
from scripts.run_cpu_pipeline import upstream_state
from sae.train_classifier import build_parser


def test_nothing_baseline_and_invalid_predictions():
    labels = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    result = accuracy_report(np.zeros_like(labels), labels)
    assert result['accr'] == .25
    assert result['segment_accuracy'] == .75
    assert result['event_accuracy'] == 0
    assert accuracy_report(labels, labels)['macro_f1'] == 1
    assert accuracy_report(np.full_like(labels, 4), labels)['accr'] == 0
    with pytest.raises(ValueError):
        accuracy_report(labels[:, :2], labels[:, :2])


@pytest.mark.parametrize('state,exit_code,expected', [
    ('RUNNING', '0:0', False), ('COMPLETED', '0:0', True),
    ('FAILED', '1:0', None), ('TIMEOUT', '0:0', None), ('COMPLETED', '1:0', None)])
def test_upstream_fails_closed(monkeypatch, state, exit_code, expected):
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: f'123|{state}|{exit_code}|\n')
    if expected is None:
        with pytest.raises(RuntimeError):
            upstream_state('123')
    else:
        assert upstream_state('123') is expected


def test_mlp_selection_opt_in():
    args = ['--sae-checkpoint', '/tmp/s', '--train-cache', '/tmp/t', '--valid-cache', '/tmp/v',
            '--train-attrs', '/tmp/a', '--valid-attrs', '/tmp/b', '--output-dir', '/tmp/o']
    assert build_parser().parse_args(args).selection_metric == 'auto'
    assert build_parser().parse_args(args + ['--selection-metric', 'accr']).selection_metric == 'accr'


def test_cnn_training_and_prediction_on_cpu(tmp_path, monkeypatch):
    import sys
    import torch
    from contsg.eval.metrics.segment import PeakValleyClassifier1D
    from scripts.train_cpu_pipeline_cnn import main
    from scripts.cpu_pipeline_metrics import predict_cnn
    torch.set_num_threads(1)
    data = tmp_path / 'data'
    data.mkdir()
    x = np.random.default_rng(42).normal(size=(12, 128, 1)).astype(np.float32)
    y = np.tile(np.arange(4), 9).reshape(12, 3)
    for split in ('train', 'valid'):
        np.save(data / f'{split}_ts.npy', x)
        np.save(data / f'{split}_attrs_idx.npy', y)
    out = tmp_path / 'cnn'
    monkeypatch.setattr(sys, 'argv', ['cnn', '--data-root', str(data), '--output-dir', str(out),
                                    '--epochs', '1', '--device', 'cpu'])
    main()
    model = PeakValleyClassifier1D(segment_len=43)
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True))
    pred = predict_cnn(model, x, torch.device('cpu'))
    assert pred.shape == y.shape
    assert (out / 'report.json').is_file()