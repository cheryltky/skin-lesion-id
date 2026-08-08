"""Shared fixtures."""

from __future__ import annotations

import random

import pytest

from skinlesion.config import SplitConfig
from skinlesion.data.metadata import LesionRecord


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig(
        train_fraction=0.70, val_fraction=0.15, test_fraction=0.15, random_seed=42
    )


def make_record(image_id: str, lesion_id: str, dx: str, source: str = "vidir_modern") -> LesionRecord:
    return LesionRecord(
        image_id=image_id,
        lesion_id=lesion_id,
        dx=dx,
        dx_type="histo",
        age=50.0,
        sex="male",
        localization="back",
        source=source,
    )


@pytest.fixture
def record_factory():
    return make_record


@pytest.fixture
def many_records() -> list[LesionRecord]:
    """Six diagnoses, each with 20 lesions carrying 1 to 3 images."""
    rng = random.Random(7)
    records: list[LesionRecord] = []
    image_id = 0
    lesion_id = 0

    for dx_index in range(6):
        dx = f"dx{dx_index}"
        for _ in range(20):
            lesion_id += 1
            for _ in range(rng.randint(1, 3)):
                image_id += 1
                records.append(make_record(f"ISIC_{image_id:07d}", f"HAM_{lesion_id:07d}", dx))

    return records
