"""Tests for configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skinlesion.config import (
    ConfigError,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    load_config,
)

VALID_CONFIG = Path("config/default.yaml")


def test_default_config_loads():
    config = load_config(VALID_CONFIG)

    assert "mel" in config.data.classes
    assert config.split.train_fraction == pytest.approx(0.70)
    assert config.paths.images_dir == Path("data/images")


def test_missing_file_reports_the_path(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="could not parse"):
        load_config(path)


def test_missing_section_is_named(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text(textwrap.dedent("""
        data:
          dataverse_doi: "x"
          classes: [nv, mel]
          min_images_per_class: 1
          source_filter: null
    """), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "missing the required section(s)" in message
    # Every absent section is named, not just the first one encountered.
    for section in ("split", "model", "training", "evaluation", "paths"):
        assert section in message


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        from skinlesion.config import _build_section

        _build_section(SplitConfig, {
            "train_fraction": 0.7, "val_fraction": 0.15, "test_fraction": 0.15,
            "random_seed": 1, "typo_key": 5,
        }, "split")


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ConfigError, match="must sum to 1.0"):
        SplitConfig(train_fraction=0.7, val_fraction=0.2, test_fraction=0.2, random_seed=1)


def test_split_fraction_out_of_range_is_rejected():
    with pytest.raises(ConfigError, match="strictly between 0 and 1"):
        SplitConfig(train_fraction=1.0, val_fraction=0.0, test_fraction=0.0, random_seed=1)


def test_too_few_classes_is_rejected():
    with pytest.raises(ConfigError, match="at least 2 diagnoses"):
        DataConfig(
            dataverse_doi="x", classes=("mel",), min_images_per_class=10, source_filter=None
        )


def test_non_positive_min_images_per_class_is_rejected():
    with pytest.raises(ConfigError, match="min_images_per_class"):
        DataConfig(
            dataverse_doi="x", classes=("nv", "mel"), min_images_per_class=0, source_filter=None
        )


def test_warmup_cannot_exceed_total_epochs():
    with pytest.raises(ConfigError, match="warmup_frozen_epochs cannot exceed"):
        TrainingConfig(
            epochs=3, batch_size=8, learning_rate=1e-3, weight_decay=0.0,
            warmup_frozen_epochs=5, label_smoothing=0.0, num_workers=0,
            early_stopping_patience=2,
        )


def test_abstain_threshold_must_be_a_probability():
    with pytest.raises(ConfigError, match="abstain_threshold"):
        EvaluationConfig(
            top_k=3, calibration_bins=10, abstain_threshold=1.5, malignant_classes=("mel",)
        )


def test_malignant_classes_must_not_be_empty():
    with pytest.raises(ConfigError, match="malignant_classes"):
        EvaluationConfig(
            top_k=3, calibration_bins=10, abstain_threshold=0.5, malignant_classes=()
        )


def test_dropout_must_be_below_one():
    with pytest.raises(ConfigError, match="dropout"):
        ModelConfig(backbone="resnet18", pretrained=True, image_size=224, dropout=1.0)


def test_with_overrides_does_not_mutate_the_original():
    config = load_config(VALID_CONFIG)
    original_epochs = config.training.epochs

    from dataclasses import replace

    modified = config.with_overrides(
        training=replace(config.training, epochs=original_epochs + 5)
    )

    assert config.training.epochs == original_epochs
    assert modified.training.epochs == original_epochs + 5


def test_with_overrides_rejects_unknown_sections():
    config = load_config(VALID_CONFIG)

    with pytest.raises(ConfigError, match="unknown config sections"):
        config.with_overrides(nonsense={})
