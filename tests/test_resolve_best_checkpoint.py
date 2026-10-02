import json

import pytest

from sae.resolve_best_checkpoint import resolve_best_checkpoint


def test_resolves_recorded_absolute_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "finetune" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (tmp_path / "summary.json").write_text(json.dumps({
        "best_checkpoint": str(checkpoint),
    }))

    assert resolve_best_checkpoint(tmp_path) == checkpoint.resolve()


def test_rejects_experiment_without_completion_summary(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed experiment summary"):
        resolve_best_checkpoint(tmp_path)


def test_rejects_missing_recorded_checkpoint(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "best_checkpoint": str(tmp_path / "missing.ckpt"),
    }))

    with pytest.raises(FileNotFoundError, match="recorded best checkpoint"):
        resolve_best_checkpoint(tmp_path)