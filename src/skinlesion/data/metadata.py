"""Read the HAM10000 metadata file into typed, immutable records.

Two things matter here beyond simply parsing the TSV:

1.  Every image keeps its lesion id. HAM10000 frequently has several
    dermatoscopic images of one lesion, so splits must be grouped by lesion
    to avoid leakage -- the direct analogue of grouping by observation in
    the nudibranch project. See data.splits.
2.  Every image keeps its source dataset (`vidir_modern`, `vidir_molemax`,
    `vienna_dias`, `rosendahl`). The `rosendahl` source is the one
    identifiably Australian slice: images from a University of
    Queensland-affiliated skin cancer practice, ethics-approved and 100%
    histopathologically confirmed. Keeping the source visible per record
    lets evaluation report on that slice separately rather than only as an
    unlabelled part of the whole.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DIAGNOSIS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}


class MetadataError(ValueError):
    """Raised when the metadata file is missing or malformed."""


@dataclass(frozen=True)
class LesionRecord:
    """One dermatoscopic image, tied to the lesion it came from."""

    image_id: str
    lesion_id: str
    dx: str
    dx_type: str
    age: float | None
    sex: str | None
    localization: str | None
    source: str

    @property
    def filename(self) -> str:
        return f"{self.image_id}.jpg"

    @property
    def is_australian(self) -> bool:
        """The Rosendahl (Queensland) subset -- see module docstring."""
        return self.source == "rosendahl"


def read_metadata(
    path: Path,
    allowed_classes: Sequence[str] | None = None,
    source_filter: str | None = None,
) -> tuple[LesionRecord, ...]:
    """Parse HAM10000_metadata.tab into LesionRecord tuples.

    Rows whose `dx` is not in allowed_classes are skipped when a filter is
    given; the default of None keeps all seven diagnoses. source_filter
    restricts to one contributing dataset -- "rosendahl" for the
    Queensland-sourced, histopathologically-confirmed subset.
    """
    if not path.is_file():
        raise MetadataError(
            f"metadata file not found: {path}. Run scripts/fetch_data.py first."
        )

    allowed = set(allowed_classes) if allowed_classes is not None else None
    records: list[LesionRecord] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise MetadataError(f"metadata file is missing column(s): {sorted(missing)}")

        for row in reader:
            dx = row["dx"].strip('"')
            if allowed is not None and dx not in allowed:
                continue
            source = row["dataset"].strip('"')
            if source_filter is not None and source != source_filter:
                continue

            age_raw = row["age"].strip('"')
            try:
                age = float(age_raw) if age_raw and age_raw.lower() != "nan" else None
            except ValueError:
                age = None

            records.append(
                LesionRecord(
                    image_id=row["image_id"].strip('"'),
                    lesion_id=row["lesion_id"].strip('"'),
                    dx=dx,
                    dx_type=row["dx_type"].strip('"'),
                    age=age,
                    sex=row["sex"].strip('"') or None,
                    localization=row["localization"].strip('"') or None,
                    source=row["dataset"].strip('"'),
                )
            )

    if not records:
        raise MetadataError(
            f"{path} contained no rows matching classes={allowed_classes} "
            f"source={source_filter}"
        )

    return tuple(records)


def drop_sparse_classes(
    records: Sequence[LesionRecord], min_images_per_class: int
) -> tuple[tuple[LesionRecord, ...], tuple[str, ...]]:
    """Drop diagnoses with fewer than min_images_per_class images.

    The direct analogue of min_observations_per_species in the nudibranch
    project: a class with only a handful of images cannot be modelled or
    evaluated sensibly, so it is excluded up front with a clear reason
    rather than left to fail unpredictably later in the split or in
    training. Returns the filtered records and the names of dropped
    classes, in that order.
    """
    if min_images_per_class < 1:
        raise ValueError("min_images_per_class must be positive")

    counts = Counter(record.dx for record in records)
    dropped = tuple(sorted(dx for dx, count in counts.items() if count < min_images_per_class))
    if not dropped:
        return tuple(records), ()

    kept = tuple(record for record in records if record.dx not in dropped)
    return kept, dropped
