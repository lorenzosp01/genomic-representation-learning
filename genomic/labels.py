"""Deterministic, task-specific label mappings shared across models.

The primary breed-classification task maps the 34 primary breeds to a dense
``0..33`` integer vocabulary in sorted-alphabetical order. The same breed must
receive the same integer label in the VMGP model, the contrastive model and the
classical baselines.

Future RQ2 uses a *separate* mapper instance over its own fixed 15-breed subset,
producing a dense ``0..14`` vocabulary (never a sparse 34-class output).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


class CanonicalLabelMapper:
    """Sorted-alphabetical vocabulary -> dense ``0..K-1`` integer labels.

    Instances are immutable once built. Construction is deterministic: the
    vocabulary is the sorted set of unique breeds, so a given breed always maps
    to the same integer regardless of call order.
    """

    def __init__(self, vocab: Sequence[str]):
        self._vocab: tuple = tuple(vocab)
        if len(set(self._vocab)) != len(self._vocab):
            raise ValueError("Label vocabulary contains duplicate entries.")
        self._index = {b: i for i, b in enumerate(self._vocab)}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_breeds(cls, breeds: Iterable[str]) -> "CanonicalLabelMapper":
        """Build a mapper over the sorted unique breeds in ``breeds``."""
        return cls(sorted(set(breeds)))

    def restrict(self, breeds: Iterable[str]) -> "CanonicalLabelMapper":
        """Return a new dense mapper over the breeds that are in both this
        vocabulary and ``breeds`` (order follows this vocabulary's order).
        """
        keep = set(breeds)
        return CanonicalLabelMapper([b for b in self._vocab if b in keep])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self._vocab)

    @property
    def classes(self) -> List[str]:
        return list(self._vocab)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, breeds) -> np.ndarray:
        """Map breed strings (list/array) to dense integer labels."""
        breeds = np.asarray(breeds)
        flat = breeds.reshape(-1)
        idx = np.empty(flat.shape, dtype=np.int64)
        for i, b in enumerate(flat):
            if b not in self._index:
                raise ValueError(
                    f"Breed {b!r} is not in the canonical vocabulary "
                    f"({self.n_classes} classes)."
                )
            idx[i] = self._index[b]
        return idx.reshape(breeds.shape)

    def inverse(self, labels) -> np.ndarray:
        """Map dense integer labels back to breed strings (object array)."""
        labels = np.asarray(labels, dtype=np.int64)
        return np.array([self._vocab[i] for i in labels.reshape(-1)],
                        dtype=object).reshape(labels.shape)

    def __len__(self) -> int:
        return len(self._vocab)

    def __repr__(self) -> str:
        return f"CanonicalLabelMapper(n_classes={self.n_classes})"
