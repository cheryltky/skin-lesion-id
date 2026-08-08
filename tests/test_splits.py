"""Tests for lesion-grouped splitting.

The leakage guarantee is the reason this module exists, so it is tested
directly rather than inferred from accuracy numbers.
"""

from __future__ import annotations

import pytest

from skinlesion.data.splits import (
    SPLIT_NAMES,
    DatasetSplit,
    SplitError,
    assert_no_leakage,
    build_splits,
)


def test_no_lesion_appears_in_two_splits(many_records, split_config):
    split = build_splits(many_records, split_config)

    train = split.lesion_ids("train")
    val = split.lesion_ids("val")
    test = split.lesion_ids("test")

    assert train & val == frozenset()
    assert train & test == frozenset()
    assert val & test == frozenset()


def test_assert_no_leakage_passes_on_a_valid_split(many_records, split_config):
    assert_no_leakage(build_splits(many_records, split_config))


def test_assert_no_leakage_detects_a_shared_lesion(record_factory):
    shared = record_factory("ISIC_0000001", "HAM_0000100", "dx0")
    leaked = DatasetSplit(
        train=(shared,),
        val=(record_factory("ISIC_0000002", "HAM_0000100", "dx0"),),
        test=(record_factory("ISIC_0000003", "HAM_0000300", "dx0"),),
        class_names=("dx0",),
    )

    with pytest.raises(SplitError, match="appear in both train and val"):
        assert_no_leakage(leaked)


def test_every_image_is_assigned_exactly_once(many_records, split_config):
    split = build_splits(many_records, split_config)

    assigned = [r.image_id for name in SPLIT_NAMES for r in getattr(split, name)]

    assert len(assigned) == len(many_records)
    assert len(set(assigned)) == len(assigned)


def test_every_class_appears_in_every_split(many_records, split_config):
    split = build_splits(many_records, split_config)

    for name in SPLIT_NAMES:
        present = {r.dx for r in getattr(split, name)}
        assert present == set(split.class_names), f"{name} is missing classes"


def test_split_is_deterministic(many_records, split_config):
    first = build_splits(many_records, split_config)
    second = build_splits(many_records, split_config)

    assert first.lesion_ids("test") == second.lesion_ids("test")


def test_split_proportions_are_roughly_respected(many_records, split_config):
    split = build_splits(many_records, split_config)
    total = sum(split.counts().values())

    train_share = len(split.train) / total
    # Grouping by lesion makes exact fractions impossible, so allow a
    # reasonable tolerance around the 0.70 target.
    assert 0.55 < train_share < 0.85


def test_class_with_too_few_lesions_is_dropped(record_factory, split_config):
    records = (
        [record_factory(f"ISIC_{i:07d}", f"HAM_{i:07d}", "common") for i in range(1, 31)]
        + [
            record_factory("ISIC_0000100", "HAM_0000100", "rare"),
            record_factory("ISIC_0000101", "HAM_0000100", "rare"),
        ]
        + [record_factory(f"ISIC_{i:07d}", f"HAM_{i:07d}", "other") for i in range(200, 230)]
    )

    split = build_splits(records, split_config)

    # "rare" has 2 images but only one lesion, so it cannot populate
    # three splits.
    assert "rare" not in split.class_names
    assert "common" in split.class_names


def test_empty_input_is_rejected(split_config):
    with pytest.raises(SplitError, match="no records supplied"):
        build_splits([], split_config)


def test_single_usable_class_is_rejected(record_factory, split_config):
    records = [record_factory(f"ISIC_{i:07d}", f"HAM_{i:07d}", "only") for i in range(1, 11)]

    with pytest.raises(SplitError, match="at least two classes"):
        build_splits(records, split_config)
