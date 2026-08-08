"""Train the classifier and write a checkpoint.

Assumes scripts/fetch_data.py has already downloaded and extracted the
images. Fits the calibration temperature on the validation split and
stores it in the checkpoint so the app reports honest confidence.

Usage:
    python scripts/train.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from skinlesion.config import ConfigError, load_config
from skinlesion.data.datasets import (
    LesionImageDataset,
    build_eval_transform,
    build_train_transform,
)
from skinlesion.data.metadata import drop_sparse_classes, read_metadata
from skinlesion.data.splits import assert_no_leakage, build_splits
from skinlesion.evaluation.calibration import fit_temperature
from skinlesion.modelling.model import (
    CheckpointMetadata,
    build_model,
    save_checkpoint,
    select_device,
)
from skinlesion.modelling.train import train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--device", type=str, default=None, help="Force cpu, mps or cuda.")
    return parser.parse_args(argv)


def collect_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            output = model(images.to(device))
            logits.append(output.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(logits), np.concatenate(labels)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        records = read_metadata(
            config.paths.metadata_file,
            allowed_classes=config.data.classes,
            source_filter=config.data.source_filter,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not read the metadata: {exc}", file=sys.stderr)
        return 1

    records, dropped = drop_sparse_classes(records, config.data.min_images_per_class)
    if dropped:
        print(f"Dropping classes below min_images_per_class: {list(dropped)}")

    split = build_splits(records, config.split)
    assert_no_leakage(split)
    print(
        f"Split: {split.counts()} images over {len(split.class_names)} diagnoses. "
        "No lesion crosses splits."
    )

    try:
        train_set = LesionImageDataset(
            split.train,
            config.paths.images_dir,
            split.class_to_index,
            build_train_transform(config.model.image_size),
        )
        val_set = LesionImageDataset(
            split.val,
            config.paths.images_dir,
            split.class_to_index,
            build_eval_transform(config.model.image_size),
        )
    except ValueError as exc:
        print(f"Could not build datasets: {exc}", file=sys.stderr)
        return 1

    if train_set.missing_files:
        print(f"Warning: {train_set.missing_files} training images are missing from disk.")

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    device = select_device(args.device)
    print(f"Training on {device}")

    model = build_model(config.model, num_classes=len(split.class_names))
    history = train_model(
        model,
        train_loader,
        val_loader,
        config.training,
        device,
        class_weights=train_set.class_weights(len(split.class_names)),
        on_progress=print,
    )

    print("\nFitting the calibration temperature on the validation split...")
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"  temperature {temperature:.3f}")

    checkpoint_path = config.paths.artifacts_dir / "model.pt"
    save_checkpoint(
        model,
        CheckpointMetadata(
            backbone=config.model.backbone,
            image_size=config.model.image_size,
            class_names=split.class_names,
            val_accuracy=history.best_val_accuracy,
            temperature=temperature,
        ),
        checkpoint_path,
    )

    history_path = config.paths.artifacts_dir / "history.json"
    history_path.write_text(json.dumps(history.as_rows(), indent=2), encoding="utf-8")

    print(
        f"\nBest validation accuracy {history.best_val_accuracy:.4f} at epoch "
        f"{history.best_epoch}\nCheckpoint: {checkpoint_path}\nHistory: {history_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
