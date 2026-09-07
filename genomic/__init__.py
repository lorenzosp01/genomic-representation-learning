"""Model-independent genomic utilities for the thesis experimental framework."""

from genomic.splitting import (
    PROTOCOL_VERSION,
    SplitIndex,
    build_split,
    load_split,
    save_split,
)

__all__ = [
    "PROTOCOL_VERSION",
    "SplitIndex",
    "build_split",
    "load_split",
    "save_split",
]
