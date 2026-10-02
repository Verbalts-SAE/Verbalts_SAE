"""Isolated CLI training entry with train-only Bridge exemplar conditioning."""
from __future__ import annotations

import numpy as np
import torch


def exemplar_index(index: int, n_train: int) -> int:
    if n_train <= 0:
        raise ValueError("empty training exemplar pool")
    return (int(index) * 104729 + 42) % n_train


def install_train_only_exemplars() -> None:
    # Process-local patch: no changes to shared modules or existing jobs.
    from contsg.data.datamodule import TimeSeriesDataset

    original = TimeSeriesDataset.__getitem__

    def getitem(self, index):
        item = original(self, index)
        if self.provide_bridge_example:
            if self.normalize:
                raise ValueError("electricity exemplar protocol requires normalize=false")
            if not hasattr(self, "_train_exemplar_pool"):
                self._train_exemplar_pool = np.load(
                    self.data_folder / "train_ts.npy", mmap_mode="r")
            pool = self._train_exemplar_pool
            row = np.array(pool[exemplar_index(index, len(pool))], dtype=np.float32, copy=True)
            if row.ndim == 1:
                row = row[:, None]
            if tuple(row.shape) != tuple(item["ts"].shape):
                raise ValueError("exemplar/target shape mismatch")
            item["bridge_example_ts"] = torch.from_numpy(row)
        return item

    TimeSeriesDataset.__getitem__ = getitem


if __name__ == "__main__":
    install_train_only_exemplars()
    from contsg.cli import app
    app()