#!/usr/bin/env python
"""Wrapper for sae.evaluate_diffusets_steering that additionally allows --split test.

The upstream CLI restricts --split to ("valid",) although all data loading is
already parameterized by split name.  This wrapper patches the argparse action
at runtime and defers to the upstream main() unchanged.
"""
import sys

import sae.evaluate_diffusets_steering as _mod

_original = _mod.build_parser


def _patched_build_parser():
    parser = _original()
    for action in parser._actions:
        if action.dest == "split":
            action.choices = ("valid", "test")
    return parser


_mod.build_parser = _patched_build_parser

if __name__ == "__main__":
    sys.exit(_mod.main())
