"""Tests for deterministic, task-specific label mappings."""

import numpy as np
import pytest

from genomic.labels import CanonicalLabelMapper


def test_sorted_alphabetical_vocab():
    mapper = CanonicalLabelMapper.from_breeds(["ZEB", "ALP", "B", "ALP", "ANG"])
    assert mapper.classes == ["ALP", "ANG", "B", "ZEB"]
    assert mapper.n_classes == 4


def test_dense_encode_deterministic():
    breeds = ["C", "A", "B", "A", "C"]
    m1 = CanonicalLabelMapper.from_breeds(breeds)
    m2 = CanonicalLabelMapper.from_breeds(breeds)
    assert np.array_equal(m1.encode(breeds), m2.encode(breeds))
    assert m1.encode(["A", "B", "C"]).tolist() == [0, 1, 2]


def test_same_breed_same_label_across_instances():
    a = CanonicalLabelMapper.from_breeds(["B", "A"])
    b = CanonicalLabelMapper.from_breeds(["A", "B"])
    assert a.encode(["A"]).tolist() == b.encode(["A"]).tolist() == [0]


def test_inverse_roundtrip():
    breeds = ["ZEB", "ALP", "ANG", "ALP"]
    mapper = CanonicalLabelMapper.from_breeds(breeds)
    labels = mapper.encode(breeds)
    back = mapper.inverse(labels)
    assert back.tolist() == breeds


def test_encode_array_preserves_shape():
    breeds = np.array([["A", "B"], ["C", "A"]], dtype=object)
    mapper = CanonicalLabelMapper.from_breeds(breeds.reshape(-1))
    labels = mapper.encode(breeds)
    assert labels.shape == (2, 2)
    assert mapper.inverse(labels).tolist() == breeds.tolist()


def test_unknown_breed_raises():
    mapper = CanonicalLabelMapper.from_breeds(["A", "B"])
    with pytest.raises(ValueError, match="not in the canonical vocabulary"):
        mapper.encode(["A", "Z"])


def test_duplicate_vocab_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalLabelMapper(["A", "A"])


def test_restrict_rq2_subset_mapping():
    primary = CanonicalLabelMapper.from_breeds(["A", "B", "C", "D"])
    rq2 = primary.restrict(["C", "A"])
    # RQ2 gets its own dense 0..1 vocabulary, not the sparse primary indices.
    assert rq2.classes == ["A", "C"]
    assert rq2.encode(["A", "C"]).tolist() == [0, 1]
    assert primary.encode(["A", "C"]).tolist() == [0, 2]


def test_primary_34_breed_protocol_shape():
    # The real 34 primary breeds are mapped by sorting them; this just pins the
    # generic behavior with the known vocabulary size.
    breeds = [f"BR{i:02d}" for i in range(34)]
    mapper = CanonicalLabelMapper.from_breeds(breeds)
    assert mapper.n_classes == 34
    assert mapper.encode(["BR00"]).tolist() == [0]
    assert mapper.encode(["BR33"]).tolist() == [33]
