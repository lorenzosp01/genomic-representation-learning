"""Pytest configuration: make the repository root importable.

Ensures ``import genomic`` resolves to the local ``genomic/`` package when
running ``pytest`` from the repository root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
