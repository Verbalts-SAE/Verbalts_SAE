#!/usr/bin/env python
"""Wrapper for sae.evaluate_diffusets_steering with legacy captions + --split test.

Main protocol (2026-09-25, user decision): legacy caption line (prev_synthustyle).
Two monkey-patches:
1. allow --split test (upstream restricts to valid);
2. redirect {split}_text_caps.npy / {split}_cap_emb.npy loads to the
   *.prev_synthustyle.npy variants so conditions, targets and audit hashes
   all refer to the legacy-caption files.
"""
import sys
from pathlib import Path

import numpy as np

import sae.evaluate_diffusets_steering as _mod
import sae.provenance as _prov

_original_build_parser = _mod.build_parser


def _patched_build_parser():
    parser = _original_build_parser()
    for action in parser._actions:
        if action.dest == "split":
            action.choices = ("valid", "test")
    return parser


_mod.build_parser = _patched_build_parser

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
