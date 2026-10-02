#!/usr/bin/env python
"""Run sae.evaluate_bridge_steering with the LEGACY caption files (prev_synthustyle).

Main protocol (2026-09-25, user decision): legacy caption line.
The upstream evaluator hardcodes {split}_text_caps.npy / {split}_cap_emb.npy under
--data-root. This wrapper transparently redirects those loads to the
*.prev_synthustyle.npy variants, so generation conditions, shape targets and
audit hashes all refer to the legacy-caption files.
"""
import sys
from pathlib import Path

import numpy as np

import sae.evaluate_bridge_steering as _mod
import sae.provenance as _prov

LEGACY = ".prev_synthustyle"

_orig_np_load = np.load


def _redirect(path):
    p = Path(str(path))
    if p.name.endswith(("_text_caps.npy", "_cap_emb.npy")):
        legacy = p.with_name(p.name[:-4] + LEGACY + ".npy")
        if legacy.is_file():
            return legacy
    return p


def _patched_load(file, *args, **kwargs):
    return _orig_np_load(_redirect(file), *args, **kwargs)


_orig_sha = _prov.sha256_file


def _patched_sha(path, *args, **kwargs):
    return _orig_sha(_redirect(path), *args, **kwargs)


np.load = _patched_load
_prov.sha256_file = _patched_sha
_mod.sha256_file = _patched_sha

if __name__ == "__main__":
    sys.exit(_mod.main())
