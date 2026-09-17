#!/usr/bin/env python
"""Companion entry point for RQ4 (marker-selection strategy).

RQ3 (marker efficiency vs panel size) and RQ4 (which ranking strategy finds
informative panels) are two readings of the same zero-leakage experiment, so
this script is an alias for ``scripts/run_rq3.py``. Both write the same frozen
artifacts to ``results/rq3_marker_efficiency/`` and
``results/rq4_marker_ranking/``.

Usage:
    python scripts/run_rq4.py                 # same as scripts/run_rq3.py
    python scripts/run_rq4.py --smoke         # tiny synthetic dry-run, K=[10,20]
"""

from __future__ import annotations

import os
import sys

# When executed directly, sys.path[0] is the scripts/ directory, so the shared
# pipeline module is importable by its own name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_rq3 import main


if __name__ == "__main__":
    raise SystemExit(main())
