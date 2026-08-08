"""Typed, immutable configuration loaded from YAML.

Every stage of the pipeline reads its settings from here so that a run is
fully described by a single config file. Values are validated on load, so
a bad config fails immediately with a clear message rather than surfacing
as a confusing error deep inside training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class ConfigError(ValueError):
    """Raised when a configuration file is missing values or self-inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    """Which diagnoses to train on and where the source files live."""

    dataverse_doi: str
    classes: tuple[str, ...]
    min_images_per_class: int
    source_filter: str | None

    def __post_init__(self) -> None:
        if len(self.classes) < 2:
            raise ConfigError("data.classes must list at least 2 diagnoses")
        if self.min_images_per_class < 1:
            raise ConfigError("data.min_images_per_class must be positive")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_seed: int

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split fractions must sum to 1.0, got {total:.6f}")
        for name in ("train_fraction", "val_fraction", "test_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"split.{name} must lie strictly between 0 and 1, got {value}")


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool
    image_size: int
    dropout: float

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ConfigError(f"model.image_size must be at least 32, got {self.image_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must lie in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_frozen_epochs: int
    label_smoothing: float
    num_workers: int
    early_stopping_patience: int

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError("training.epochs must be at least 1")
        if self.batch_size < 1:
            raise ConfigError("training.batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be positive")
        if self.warmup_frozen_epochs > self.epochs:
            raise ConfigError(
                "training.warmup_frozen_epochs cannot exceed training.epochs, "
                f"got {self.warmup_frozen_epochs} > {self.epochs}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError("training.label_smoothing must lie in [0, 1)")


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int
    calibration_bins: int
    abstain_threshold: float
    malignant_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError("evaluation.top_k must be at least 1")
        if self.calibration_bins < 2:
            raise ConfigError("evaluation.calibration_bins must be at least 2")
        if not 0.0 <= self.abstain_threshold <= 1.0:
            raise ConfigError("evaluation.abstain_threshold must lie in [0, 1]")
        if not self.malignant_classes:
            raise ConfigError(
                "evaluation.malignant_classes must list at least one diagnosis code"
            )


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    raw_dir: Path
    images_dir: Path
    metadata_file: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class Config:
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig

    def with_overrides(self, **sections: Any) -> "Config":
        """Return a new Config with whole sections replaced.

        The original is never mutated, so a notebook can branch a config
        for an experiment without disturbing the loaded baseline.
        """
        unknown = set(sections) - set(self.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown config sections: {sorted(unknown)}")
        return replace(self, **sections)


def _require(mapping: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    if section not in mapping:
        raise ConfigError(f"config is missing the required '{section}' section")
    value = mapping[section]
    if not isinstance(value, Mapping):
        raise ConfigError(f"config section '{section}' must be a mapping, got {type(value).__name__}")
    return value


def _build_section(cls: type, values: Mapping[str, Any], section: str) -> Any:
    expected = set(cls.__dataclass_fields__)
    provided = set(values)

    missing = expected - provided
    if missing:
        raise ConfigError(f"config section '{section}' is missing keys: {sorted(missing)}")

    unexpected = provided - expected
    if unexpected:
        raise ConfigError(f"config section '{section}' has unknown keys: {sorted(unexpected)}")

    return cls(**dict(values))


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate configuration from a YAML file.

    Raises ConfigError with an actionable message if the file is missing,
    unparseable, or fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path} as YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    required = ("data", "split", "model", "training", "evaluation", "paths")
    absent = [section for section in required if section not in raw]
    if absent:
        raise ConfigError(
            f"{config_path} is missing the required section(s): {', '.join(absent)}"
        )

    data_raw = dict(_require(raw, "data"))
    if isinstance(data_raw.get("classes"), list):
        data_raw["classes"] = tuple(data_raw["classes"])

    evaluation_raw = dict(_require(raw, "evaluation"))
    if isinstance(evaluation_raw.get("malignant_classes"), list):
        evaluation_raw["malignant_classes"] = tuple(evaluation_raw["malignant_classes"])

    paths_raw = {key: Path(value) for key, value in _require(raw, "paths").items()}

    return Config(
        data=_build_section(DataConfig, data_raw, "data"),
        split=_build_section(SplitConfig, _require(raw, "split"), "split"),
        model=_build_section(ModelConfig, _require(raw, "model"), "model"),
        training=_build_section(TrainingConfig, _require(raw, "training"), "training"),
        evaluation=_build_section(EvaluationConfig, evaluation_raw, "evaluation"),
        paths=_build_section(PathsConfig, paths_raw, "paths"),
    )
