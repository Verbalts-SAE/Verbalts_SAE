"""Resolve a completed ConTSG experiment's recorded best checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_best_checkpoint(experiment_dir: Path) -> Path:
    """Return the recorded best checkpoint, rejecting incomplete experiments."""
    experiment_dir = experiment_dir.expanduser().resolve()
    summary_path = experiment_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"completed experiment summary not found: {summary_path}"
        )

    summary = json.loads(summary_path.read_text())
    checkpoint_value = summary.get("best_checkpoint")
    if not checkpoint_value:
        raise ValueError(f"best_checkpoint missing from {summary_path}")

    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = experiment_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recorded best checkpoint not found: {checkpoint}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    print(resolve_best_checkpoint(args.experiment_dir))


if __name__ == "__main__":
    main()