"""Split images into train, validation and test without leaking.

HAM10000 frequently carries several dermatoscopic images of the same
lesion -- follow-up shots, different angles, sometimes the same photo
resubmitted under a different image id. Splitting at the image level puts
near-duplicates of one lesion on both sides of the split, and the
resulting accuracy is inflated and meaningless -- the exact failure mode
splitting by iNaturalist observation guards against in the nudibranch
project.

Every image from one lesion therefore lands in exactly one split.
Assignment is stratified by diagnosis and deterministic given a seed, and
assert_no_leakage makes the guarantee checkable rather than assumed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..config import SplitConfig
from .metadata import LesionRecord

SPLIT_NAMES = ("train", "val", "test")


class SplitError(ValueError):
    """Raised when a valid split cannot be produced from the given images."""


@dataclass(frozen=True)
class DatasetSplit:
    """Images assigned to each split, plus the class vocabulary."""

    train: tuple[LesionRecord, ...]
    val: tuple[LesionRecord, ...]
    test: tuple[LesionRecord, ...]
    class_names: tuple[str, ...]

    @property
    def class_to_index(self) -> Mapping[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    def counts(self) -> Mapping[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def lesion_ids(self, split: str) -> frozenset[str]:
        records = getattr(self, split)
        return frozenset(record.lesion_id for record in records)


def _group_by_lesion(records: Iterable[LesionRecord]) -> dict[str, list[LesionRecord]]:
    grouped: dict[str, list[LesionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.lesion_id].append(record)
    return dict(grouped)


def _assign_lesions(
    lesions: Sequence[tuple[str, list[LesionRecord]]],
    config: SplitConfig,
    rng: random.Random,
) -> dict[str, list[LesionRecord]]:
    """Greedily assign whole lesions to hit the target image fractions.

    Lesions are shuffled, then each is placed in whichever split is
    furthest below its target share. Greedy placement handles the uneven
    image counts per lesion better than slicing a shuffled list.
    """
    shuffled = list(lesions)
    rng.shuffle(shuffled)

    targets = {
        "train": config.train_fraction,
        "val": config.val_fraction,
        "test": config.test_fraction,
    }
    assigned: dict[str, list[LesionRecord]] = {name: [] for name in SPLIT_NAMES}

    # Seed one lesion into each split first so no split is ever empty for
    # this diagnosis, which would drop the class from validation or test.
    for name in SPLIT_NAMES:
        if not shuffled:
            break
        _, records = shuffled.pop()
        assigned[name].extend(records)

    for _, records in shuffled:
        total = sum(len(group) for group in assigned.values()) or 1
        deficits = {
            name: targets[name] - (len(assigned[name]) / total) for name in SPLIT_NAMES
        }
        chosen = max(SPLIT_NAMES, key=lambda name: deficits[name])
        assigned[chosen].extend(records)

    return assigned


def build_splits(records: Sequence[LesionRecord], config: SplitConfig) -> DatasetSplit:
    """Build a stratified, lesion-grouped train/val/test split.

    Diagnoses with fewer than three distinct lesions are dropped, since
    they cannot appear in all three splits and would otherwise produce a
    class the model can never be evaluated on.
    """
    if not records:
        raise SplitError("no records supplied")

    by_dx: dict[str, list[LesionRecord]] = defaultdict(list)
    for record in records:
        by_dx[record.dx].append(record)

    buckets: dict[str, list[LesionRecord]] = {name: [] for name in SPLIT_NAMES}
    kept_classes: list[str] = []
    dropped: list[str] = []

    for dx in sorted(by_dx):
        grouped = _group_by_lesion(by_dx[dx])
        if len(grouped) < len(SPLIT_NAMES):
            dropped.append(dx)
            continue

        kept_classes.append(dx)
        # Seed per class so adding a class does not reshuffle the others.
        rng = random.Random(f"{config.random_seed}:{dx}")
        assigned = _assign_lesions(sorted(grouped.items()), config, rng)
        for name in SPLIT_NAMES:
            buckets[name].extend(assigned[name])

    if len(kept_classes) < 2:
        raise SplitError(
            f"only {len(kept_classes)} diagnoses had at least {len(SPLIT_NAMES)} distinct "
            f"lesions. A classifier needs at least two classes. Dropped: {dropped}"
        )

    return DatasetSplit(
        train=tuple(buckets["train"]),
        val=tuple(buckets["val"]),
        test=tuple(buckets["test"]),
        class_names=tuple(kept_classes),
    )


def assert_no_leakage(split: DatasetSplit) -> None:
    """Raise if any lesion appears in more than one split.

    Cheap to run and worth running: this is the property the whole split
    strategy exists to guarantee.
    """
    train_ids = split.lesion_ids("train")
    val_ids = split.lesion_ids("val")
    test_ids = split.lesion_ids("test")

    for left_name, left, right_name, right in (
        ("train", train_ids, "val", val_ids),
        ("train", train_ids, "test", test_ids),
        ("val", val_ids, "test", test_ids),
    ):
        overlap = left & right
        if overlap:
            raise SplitError(
                f"{len(overlap)} lesion(s) appear in both {left_name} and "
                f"{right_name}, for example {sorted(overlap)[:5]}"
            )
