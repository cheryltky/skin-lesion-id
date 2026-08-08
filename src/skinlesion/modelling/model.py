"""Model construction, device selection and checkpointing.

A checkpoint carries its class names and image size alongside the weights.
Without that, the Streamlit app would need to reconstruct the label
vocabulary from the dataset, and a reordered class list would silently
mislabel every prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import timm
import torch
from torch import nn

from ..config import ModelConfig


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything needed to use a set of weights correctly."""

    backbone: str
    image_size: int
    class_names: tuple[str, ...]
    val_accuracy: float
    temperature: float

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def select_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device.

    Apple Silicon exposes MPS, which is markedly faster than CPU for this
    model size. CUDA is preferred where present.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_backbone(name: str, **kwargs) -> nn.Module:
    """Create a timm model, tolerating both CNN and ViT backbone families.

    ViT families (including DINOv2) accept `dynamic_img_size` to interpolate
    position embeddings for an input resolution other than their pretraining
    default; CNN backbones don't recognise the argument at all. Trying it
    first and falling back keeps callers agnostic to which family `name`
    belongs to.
    """
    try:
        return timm.create_model(name, dynamic_img_size=True, **kwargs)
    except TypeError:
        return timm.create_model(name, **kwargs)


def build_model(config: ModelConfig, num_classes: int) -> nn.Module:
    """Create a timm backbone with a fresh classification head."""
    if num_classes < 2:
        raise ValueError(f"need at least 2 classes, got {num_classes}")

    try:
        model = _create_backbone(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=num_classes,
            drop_rate=config.dropout,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create backbone {config.backbone!r}. "
            "Check the name against timm.list_models()."
        ) from exc

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classification head.

    Training the fresh head against a frozen backbone for the first few
    epochs stops large, randomly-initialised gradients from damaging the
    pretrained features.
    """
    classifier = model.get_classifier()
    classifier_params = {id(p) for p in classifier.parameters()}

    for parameter in model.parameters():
        if id(parameter) not in classifier_params:
            parameter.requires_grad = trainable


def save_checkpoint(
    model: nn.Module, metadata: CheckpointMetadata, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "metadata": asdict(metadata)},
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[nn.Module, CheckpointMetadata]:
    """Rebuild a model and its metadata from a checkpoint file."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Train a model first, or point "
            "the app at an existing checkpoint."
        )

    target = device or select_device()
    payload = torch.load(path, map_location=target, weights_only=False)

    try:
        raw = payload["metadata"]
        metadata = CheckpointMetadata(
            backbone=raw["backbone"],
            image_size=int(raw["image_size"]),
            class_names=tuple(raw["class_names"]),
            val_accuracy=float(raw["val_accuracy"]),
            temperature=float(raw["temperature"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"checkpoint {path} has malformed metadata: {exc}") from exc

    model = _create_backbone(
        metadata.backbone, pretrained=False, num_classes=metadata.num_classes
    )
    model.load_state_dict(payload["state_dict"])
    model.to(target)
    model.eval()

    return model, metadata


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_classes(class_names: Sequence[str]) -> str:
    return f"{len(class_names)} classes: {', '.join(class_names[:3])}..."
