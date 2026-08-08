"""Tests for the training loop.

A two-layer model over synthetic tensors keeps these fast while still
exercising warmup, early stopping and best-weight restoration.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skinlesion.config import TrainingConfig
from skinlesion.modelling.train import train_model


class TinyModel(nn.Module):
    """A minimal model exposing get_classifier, as timm models do."""

    def __init__(self, features: int = 8, classes: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(features, 16)
        self.head = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))

    def get_classifier(self) -> nn.Module:
        return self.head


def make_loader(samples: int = 40, features: int = 8, classes: int = 3) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(samples, features, generator=generator)
    y = torch.randint(0, classes, (samples,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def base_config(**overrides) -> TrainingConfig:
    defaults = dict(
        epochs=4,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_frozen_epochs=1,
        label_smoothing=0.0,
        num_workers=0,
        early_stopping_patience=10,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_history_records_every_epoch():
    history = train_model(
        TinyModel(), make_loader(), make_loader(), base_config(), torch.device("cpu")
    )

    assert len(history.epochs) == 4
    assert [e.epoch for e in history.epochs] == [1, 2, 3, 4]


def test_warmup_epochs_are_marked_as_frozen():
    history = train_model(
        TinyModel(),
        make_loader(),
        make_loader(),
        base_config(epochs=3, warmup_frozen_epochs=2),
        torch.device("cpu"),
    )

    assert [e.backbone_frozen for e in history.epochs] == [True, True, False]


def test_backbone_is_trainable_again_after_warmup():
    model = TinyModel()

    train_model(
        model,
        make_loader(),
        make_loader(),
        base_config(epochs=2, warmup_frozen_epochs=1),
        torch.device("cpu"),
    )

    assert all(p.requires_grad for p in model.backbone.parameters())


def test_best_weights_are_restored_at_the_end():
    model = TinyModel()

    history = train_model(
        model, make_loader(), make_loader(), base_config(), torch.device("cpu")
    )

    assert history.best_state is not None
    for key, value in model.state_dict().items():
        assert torch.allclose(value.cpu(), history.best_state[key])


def test_early_stopping_triggers_when_accuracy_stalls():
    # Random labels mean validation accuracy will not improve for long.
    history = train_model(
        TinyModel(),
        make_loader(),
        make_loader(),
        base_config(epochs=30, warmup_frozen_epochs=0, early_stopping_patience=2),
        torch.device("cpu"),
    )

    assert history.stopped_early
    assert len(history.epochs) < 30


def test_progress_callback_receives_messages():
    messages: list[str] = []

    train_model(
        TinyModel(),
        make_loader(),
        make_loader(),
        base_config(epochs=2),
        torch.device("cpu"),
        on_progress=messages.append,
    )

    assert any("epoch 1" in m for m in messages)
    assert any("val_acc" in m for m in messages)


def test_class_weights_are_accepted():
    history = train_model(
        TinyModel(),
        make_loader(),
        make_loader(),
        base_config(epochs=1, warmup_frozen_epochs=0),
        torch.device("cpu"),
        class_weights=torch.tensor([1.0, 2.0, 0.5]),
    )

    assert len(history.epochs) == 1


def test_empty_loader_is_reported():
    empty = DataLoader(TensorDataset(torch.empty(0, 8), torch.empty(0, dtype=torch.long)))

    with pytest.raises(RuntimeError, match="no batches"):
        train_model(TinyModel(), empty, make_loader(), base_config(), torch.device("cpu"))


def test_as_rows_is_serialisable():
    history = train_model(
        TinyModel(), make_loader(), make_loader(), base_config(epochs=2), torch.device("cpu")
    )

    rows = history.as_rows()

    assert len(rows) == 2
    assert set(rows[0]) == {"epoch", "train_loss", "val_loss", "val_accuracy", "backbone_frozen"}
