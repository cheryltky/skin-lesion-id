"""Tests for the lesion image dataset and class weighting."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from skinlesion.data.datasets import (
    LesionImageDataset,
    build_eval_transform,
    build_train_transform,
)

CLASS_TO_INDEX = {"nv": 0, "mel": 1}


def write_image(root, filename: str) -> None:
    Image.new("RGB", (64, 64), (10, 120, 200)).save(root / filename, format="JPEG")


@pytest.fixture
def images_dir(tmp_path, record_factory):
    records = [
        record_factory("ISIC_0000001", "HAM_0000010", "nv"),
        record_factory("ISIC_0000002", "HAM_0000011", "nv"),
        record_factory("ISIC_0000003", "HAM_0000012", "mel"),
    ]
    for record in records:
        write_image(tmp_path, record.filename)
    return tmp_path, records


def test_dataset_length_matches_files_present(images_dir):
    root, records = images_dir

    dataset = LesionImageDataset(records, root, CLASS_TO_INDEX, build_eval_transform(32))

    assert len(dataset) == 3
    assert dataset.missing_files == 0


def test_items_are_normalised_tensors_of_the_right_shape(images_dir):
    root, records = images_dir

    image, label = LesionImageDataset(records, root, CLASS_TO_INDEX, build_eval_transform(32))[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label in {0, 1}


def test_train_transform_produces_the_requested_size(images_dir):
    root, records = images_dir

    image, _ = LesionImageDataset(records, root, CLASS_TO_INDEX, build_train_transform(48))[0]

    assert image.shape == (3, 48, 48)


def test_missing_files_are_counted_not_fatal(images_dir, record_factory):
    root, records = images_dir
    absent = record_factory("ISIC_9999999", "HAM_9999999", "nv")

    dataset = LesionImageDataset(
        [*records, absent], root, CLASS_TO_INDEX, build_eval_transform(32)
    )

    assert len(dataset) == 3
    assert dataset.missing_files == 1


def test_records_outside_the_class_vocabulary_are_ignored(images_dir, record_factory):
    root, records = images_dir
    stranger = record_factory("ISIC_0000050", "HAM_0000050", "bkl")
    write_image(root, stranger.filename)

    dataset = LesionImageDataset(
        [*records, stranger], root, CLASS_TO_INDEX, build_eval_transform(32)
    )

    assert len(dataset) == 3


def test_no_resolvable_files_is_actionable(tmp_path, record_factory):
    with pytest.raises(ValueError, match="Run scripts/fetch_data.py"):
        LesionImageDataset(
            [record_factory("ISIC_0000001", "HAM_0000001", "nv")],
            tmp_path,
            CLASS_TO_INDEX,
            build_eval_transform(32),
        )


def test_empty_records_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="zero records"):
        LesionImageDataset([], tmp_path, CLASS_TO_INDEX, build_eval_transform(32))


def test_class_counts_reflect_the_distribution(images_dir):
    root, records = images_dir

    counts = LesionImageDataset(records, root, CLASS_TO_INDEX, build_eval_transform(32)).class_counts()

    assert counts == {0: 2, 1: 1}


def test_class_weights_favour_the_rarer_class(images_dir):
    root, records = images_dir
    dataset = LesionImageDataset(records, root, CLASS_TO_INDEX, build_eval_transform(32))

    weights = dataset.class_weights(num_classes=2)

    # Class 1 (mel) has half the samples of class 0 (nv), so twice the weight.
    assert weights[1] == pytest.approx(2 * weights[0])
    assert weights.mean() == pytest.approx(1.0)


def test_absent_class_receives_zero_weight(images_dir):
    root, records = images_dir
    dataset = LesionImageDataset(records, root, CLASS_TO_INDEX, build_eval_transform(32))

    weights = dataset.class_weights(num_classes=4)

    assert weights[2] == pytest.approx(0.0)
    assert weights[3] == pytest.approx(0.0)
