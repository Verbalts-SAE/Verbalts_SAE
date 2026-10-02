import pytest

from scripts.preview_cpu_baselines import snapshot_best


def test_snapshot_best_uses_validation_loss_not_last(tmp_path):
    stage = tmp_path / 'run/checkpoints/finetune'
    for epoch, loss in ((2, '.3'), (5, '.2')):
        path = stage / f'finetune-epoch={epoch}-val/loss={loss}.ckpt'
        path.parent.mkdir(parents=True)
        path.write_bytes(loss.encode())
    (stage / 'last.ckpt').write_bytes(b'last')
    output = tmp_path / 'snapshot.ckpt'
    source = snapshot_best(tmp_path, output)
    assert source.name == 'loss=.2.ckpt'
    assert output.read_bytes() == b'.2'


def test_snapshot_requires_finetune_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_best(tmp_path, tmp_path / 'snapshot.ckpt')