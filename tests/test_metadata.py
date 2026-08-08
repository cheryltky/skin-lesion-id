"""Tests for reading the HAM10000 metadata file."""

from __future__ import annotations

import textwrap

import pytest

from skinlesion.data.metadata import MetadataError, drop_sparse_classes, read_metadata

HEADER = "lesion_id\timage_id\tdx\tdx_type\tage\tsex\tlocalization\tdataset\n"


def _write(tmp_path, rows: str):
    path = tmp_path / "metadata.tab"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_reads_all_rows_by_default(tmp_path):
    path = _write(
        tmp_path,
        '"HAM_1"\t"ISIC_1"\t"nv"\t"histo"\t50.0\t"male"\t"back"\t"rosendahl"\n'
        '"HAM_2"\t"ISIC_2"\t"mel"\t"histo"\t60.0\t"female"\t"scalp"\t"vidir_modern"\n',
    )

    records = read_metadata(path)

    assert len(records) == 2
    assert {r.dx for r in records} == {"nv", "mel"}


def test_allowed_classes_filters_rows(tmp_path):
    path = _write(
        tmp_path,
        '"HAM_1"\t"ISIC_1"\t"nv"\t"histo"\t50.0\t"male"\t"back"\t"rosendahl"\n'
        '"HAM_2"\t"ISIC_2"\t"mel"\t"histo"\t60.0\t"female"\t"scalp"\t"vidir_modern"\n',
    )

    records = read_metadata(path, allowed_classes=("nv",))

    assert len(records) == 1
    assert records[0].dx == "nv"


def test_source_filter_isolates_the_australian_subset(tmp_path):
    path = _write(
        tmp_path,
        '"HAM_1"\t"ISIC_1"\t"nv"\t"histo"\t50.0\t"male"\t"back"\t"rosendahl"\n'
        '"HAM_2"\t"ISIC_2"\t"nv"\t"histo"\t60.0\t"female"\t"scalp"\t"vidir_modern"\n',
    )

    records = read_metadata(path, source_filter="rosendahl")

    assert len(records) == 1
    assert records[0].is_australian


def test_missing_age_becomes_none(tmp_path):
    path = _write(
        tmp_path,
        '"HAM_1"\t"ISIC_1"\t"nv"\t"histo"\tNaN\t"male"\t"back"\t"rosendahl"\n',
    )

    records = read_metadata(path)

    assert records[0].age is None


def test_missing_file_is_actionable(tmp_path):
    with pytest.raises(MetadataError, match="Run scripts/fetch_data.py"):
        read_metadata(tmp_path / "absent.tab")


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "bad.tab"
    path.write_text("lesion_id\timage_id\n\"HAM_1\"\t\"ISIC_1\"\n", encoding="utf-8")

    with pytest.raises(MetadataError, match="missing column"):
        read_metadata(path)


def test_filter_matching_nothing_is_reported(tmp_path):
    path = _write(
        tmp_path,
        '"HAM_1"\t"ISIC_1"\t"nv"\t"histo"\t50.0\t"male"\t"back"\t"rosendahl"\n',
    )

    with pytest.raises(MetadataError, match="no rows matching"):
        read_metadata(path, allowed_classes=("mel",))


def test_drop_sparse_classes_removes_only_classes_below_threshold(record_factory):
    records = (
        [record_factory(f"ISIC_{i}", f"HAM_{i}", "common") for i in range(10)]
        + [record_factory("ISIC_rare", "HAM_rare", "rare")]
    )

    kept, dropped = drop_sparse_classes(records, min_images_per_class=5)

    assert dropped == ("rare",)
    assert {r.dx for r in kept} == {"common"}


def test_drop_sparse_classes_is_a_noop_when_nothing_is_below_threshold(record_factory):
    records = [record_factory(f"ISIC_{i}", f"HAM_{i}", "common") for i in range(10)]

    kept, dropped = drop_sparse_classes(records, min_images_per_class=5)

    assert dropped == ()
    assert len(kept) == len(records)


def test_drop_sparse_classes_rejects_non_positive_threshold(record_factory):
    with pytest.raises(ValueError, match="must be positive"):
        drop_sparse_classes([record_factory("ISIC_1", "HAM_1", "x")], min_images_per_class=0)
