"""Incremental experiment-result persistence helpers.

These functions serialise a list of experiment-result dictionaries to JSON,
converting NumPy scalars to native Python types, so results can be recovered
after a crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Union


def save_results_incrementally(
    results_list: List[dict],
    path: Union[str, Path] = "results_cache.json",
) -> None:
    """Save a list of result dicts to a JSON file (NumPy-safe).

    Converts NumPy scalars (and lists/tuples of them) to native Python types
    before serialisation. Prints a confirmation line, matching the original
    behaviour.
    """
    path = Path(path)
    clean_results: List[dict] = []
    for res in results_list:
        clean_res = {}
        for k, v in res.items():
            if hasattr(v, "item"):  # numpy scalar
                clean_res[k] = v.item()
            elif isinstance(v, (list, tuple)):
                clean_res[k] = [x.item() if hasattr(x, "item") else x for x in v]
            else:
                clean_res[k] = v
        clean_results.append(clean_res)

    with open(path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"💾 Risultati salvati in {path}")


def load_cached_results(path: Union[str, Path] = "results_cache.json") -> list:
    """Load previously cached results; return [] if the file does not exist."""
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []
