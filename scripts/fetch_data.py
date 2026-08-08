"""Download HAM10000 from Harvard Dataverse and lay it out for training.

Non-commercial academic use, per the dataset's terms on Harvard Dataverse.
Safe to re-run: files already on disk are left alone rather than
re-downloaded or re-extracted.

Usage:
    python scripts/fetch_data.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from skinlesion.config import ConfigError, load_config
from skinlesion.data.metadata import drop_sparse_classes, read_metadata

# Harvard Dataverse file ids for doi:10.7910/DVN/DBW86T, resolved once via
# the Dataverse API (GET .../api/datasets/:persistentId/?persistentId=...)
# rather than hardcoded from the DOI, since file ids are not derivable from
# the DOI itself.
_DATAVERSE_FILES = {
    "HAM10000_metadata.tab": 4338392,
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
}
_ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    return parser.parse_args(argv)


def _download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_id in _DATAVERSE_FILES.items():
        destination = raw_dir / filename
        if destination.is_file():
            print(f"  {filename} already present, skipping")
            continue
        print(f"  downloading {filename} ...")
        urlretrieve(_ACCESS_URL.format(file_id=file_id), destination)


def _extract(raw_dir: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for part in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
        archive_path = raw_dir / part
        with zipfile.ZipFile(archive_path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".jpg")]
            to_extract = [
                n for n in names if not (images_dir / Path(n).name).is_file()
            ]
            if not to_extract:
                print(f"  {part}: all {len(names)} images already extracted")
                continue
            print(f"  {part}: extracting {len(to_extract)} of {len(names)} images ...")
            for name in to_extract:
                data = archive.read(name)
                (images_dir / Path(name).name).write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print("Downloading HAM10000 from Harvard Dataverse (doi:10.7910/DVN/DBW86T) ...")
    _download(config.paths.raw_dir)

    print(f"\nExtracting into {config.paths.images_dir} ...")
    _extract(config.paths.raw_dir, config.paths.images_dir)

    print("\nValidating metadata ...")
    records = read_metadata(
        config.paths.metadata_file,
        allowed_classes=config.data.classes,
        source_filter=config.data.source_filter,
    )
    lesions = len({r.lesion_id for r in records})
    australian = sum(1 for r in records if r.is_australian)
    print(
        f"  {len(records)} images across {lesions} lesions "
        f"({australian} from the Rosendahl/Queensland subset)"
    )

    kept, dropped = drop_sparse_classes(records, config.data.min_images_per_class)
    if dropped:
        print(
            f"  {len(dropped)} class(es) below min_images_per_class="
            f"{config.data.min_images_per_class} will be dropped at train "
            f"time: {list(dropped)}"
        )

    missing = [r for r in records if not (config.paths.images_dir / r.filename).is_file()]
    if missing:
        print(f"\n{len(missing)} images listed in the metadata were not found on disk.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
