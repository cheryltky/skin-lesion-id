"""Tests for model construction and checkpointing.

A tiny backbone keeps these fast; the behaviour under test is the wiring,
not the architecture.
"""

from __future__ import annotations

import pytest
import torch

from skinlesion.config import ModelConfig
from skinlesion.modelling.model import (
    CheckpointMetadata,
    build_model,
    count_trainable_parameters,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_backbone_trainable,
)

TINY = ModelConfig(backbone="resnet18", pretrained=False, image_size=64, dropout=0.1)


@pytest.fixture
def model():
    return build_model(TINY, num_classes=4)


def test_model_outputs_one_logit_per_class(model):
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 4)


def test_fewer_than_two_classes_is_rejected():
    with pytest.raises(ValueError, match="at least 2 classes"):
        build_model(TINY, num_classes=1)


def test_unknown_backbone_is_reported_clearly():
    config = ModelConfig(
        backbone="not_a_real_backbone", pretrained=False, image_size=64, dropout=0.0
    )

    with pytest.raises(Exception, match="not_a_real_backbone|Unknown model"):
        build_model(config, num_classes=3)


def test_vit_backbone_runs_at_a_non_native_resolution():
    """ViT backbones (including DINOv2) need dynamic_img_size to accept an
    input resolution other than their pretraining default; CNN backbones
    reject the argument outright. build_model must handle both."""
    config = ModelConfig(
        backbone="vit_tiny_patch16_224", pretrained=False, image_size=64, dropout=0.0
    )

    model = build_model(config, num_classes=3)
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 3)


def test_freezing_the_backbone_leaves_the_head_trainable(model):
    set_backbone_trainable(model, trainable=False)

    frozen_total = count_trainable_parameters(model)
    head_total = sum(p.numel() for p in model.get_classifier().parameters())

    assert frozen_total == head_total
    assert frozen_total > 0


def test_unfreezing_restores_all_parameters(model):
    everything = count_trainable_parameters(model)

    set_backbone_trainable(model, trainable=False)
    set_backbone_trainable(model, trainable=True)

    assert count_trainable_parameters(model) == everything


def test_checkpoint_round_trips_weights_and_metadata(tmp_path, model):
    metadata = CheckpointMetadata(
        backbone="resnet18",
        image_size=64,
        class_names=("nv", "mel", "bcc", "bkl"),
        val_accuracy=0.83,
        temperature=1.4,
    )
    path = tmp_path / "model.pt"

    save_checkpoint(model, metadata, path)
    restored, restored_metadata = load_checkpoint(path, device=torch.device("cpu"))

    assert restored_metadata == metadata
    assert restored_metadata.num_classes == 4

    model.eval()
    batch = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(batch), restored(batch), atol=1e-6)


def test_loading_a_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        load_checkpoint(tmp_path / "absent.pt")


def test_malformed_checkpoint_metadata_is_rejected(tmp_path, model):
    path = tmp_path / "broken.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": {"backbone": "resnet18"}}, path)

    with pytest.raises(ValueError, match="malformed metadata"):
        load_checkpoint(path, device=torch.device("cpu"))


def test_select_device_honours_an_explicit_preference():
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_returns_something_usable():
    assert select_device().type in {"cpu", "mps", "cuda"}
