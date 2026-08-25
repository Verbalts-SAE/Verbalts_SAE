"""Small, dependency-free provenance helpers for SAE experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic key order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git_revision(repo_root: str | Path) -> dict[str, Any]:
    """Record the current revision and dirty state without changing the repository."""
    root = Path(repo_root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def runtime_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable runtime audit record."""
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cuda_device": cuda_name,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_revision(repo_root),
    }


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["git_revision", "runtime_metadata", "sha256_file", "tensor_state_sha256", "write_json"]