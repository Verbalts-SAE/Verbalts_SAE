from scripts.monitor_weather36 import parse_progress


def test_serial_progress():
    text = ('START candidate=0 now\n[Epoch 1/80] train_loss=0.2 val_loss=0.3 lr=1e-4 (1.6m)\n'
            'DONE candidate=0 now\nSTART candidate=1 now\n')
    result = parse_progress(text, ['a', 'b', 'c'], 'RUNNING')
    assert result['a']['status'] == 'COMPLETED'
    assert result['a']['latest_epoch']['val_loss'] == .3
    assert result['b']['status'] == 'RUNNING'
    assert result['c']['status'] == 'NOT_STARTED'


def test_failure_not_success():
    text = 'START candidate=0 now\n'
    assert parse_progress(text, ['a'], 'FAILED')['a']['status'] == 'FAILED'
    assert parse_progress(text, ['a'], 'COMPLETED')['a']['status'] == 'INCOMPLETE'


def test_multistage_history_kept():
    text = ('START candidate=0 now\n[Epoch 60/60] train_loss=0.2 val_loss=0.3 lr=1e-4\n'
            '[Epoch 1/80] train_loss=0.9 val_loss=0.8 lr=1e-4\n')
    result = parse_progress(text, ['a'], 'RUNNING')['a']
    assert len(result['epoch_history']) == 2
    assert result['latest_epoch']['total_epochs'] == 80