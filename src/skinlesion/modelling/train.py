"""Training loop with frozen warmup, class weighting and early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from .model import set_backbone_trainable

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    backbone_frozen: bool


@dataclass
class TrainingHistory:
    """Per-epoch record of the run, plus the best weights seen."""

    epochs: list[EpochResult] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    stopped_early: bool = False

    def as_rows(self) -> list[dict[str, float | int | bool]]:
        return [
            {
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "val_accuracy": e.val_accuracy,
                "backbone_frozen": e.backbone_frozen,
            }
            for e in self.epochs
        ]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one pass. Training if an optimiser is supplied, otherwise eval.

    Returns mean loss and accuracy.
    """
    training = optimiser is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimiser.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimiser.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += batch

    if seen == 0:
        raise RuntimeError("data loader yielded no batches")

    return total_loss / seen, correct / seen


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None = None,
    on_progress: ProgressCallback | None = None,
) -> TrainingHistory:
    """Fine-tune the model, returning the history and best weights.

    The backbone stays frozen for the first warmup_frozen_epochs so the
    new head can settle, then the whole network trains at a lower rate.
    Early stopping watches validation accuracy.
    """
    report = on_progress or (lambda message: None)
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=config.label_smoothing,
    )

    history = TrainingHistory()
    epochs_without_improvement = 0
    optimiser: torch.optim.Optimizer | None = None
    currently_frozen: bool | None = None

    for epoch in range(1, config.epochs + 1):
        should_freeze = epoch <= config.warmup_frozen_epochs

        if should_freeze != currently_frozen:
            set_backbone_trainable(model, trainable=not should_freeze)
            # A frozen backbone means only the head trains, so it can take a
            # higher rate; once unfrozen, drop back to avoid wrecking the
            # pretrained features.
            learning_rate = config.learning_rate * (10.0 if should_freeze else 1.0)
            optimiser = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            currently_frozen = should_freeze
            report(
                f"epoch {epoch}: backbone {'frozen' if should_freeze else 'unfrozen'}, "
                f"lr {learning_rate:.2e}"
            )

        assert optimiser is not None
        train_loss, _ = _run_epoch(model, train_loader, criterion, device, optimiser)
        val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device, None)

        history.epochs.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                backbone_frozen=should_freeze,
            )
        )
        report(
            f"epoch {epoch}/{config.epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_accuracy:.4f}"
        )

        if val_accuracy > history.best_val_accuracy:
            history.best_val_accuracy = val_accuracy
            history.best_epoch = epoch
            history.best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                history.stopped_early = True
                report(
                    f"stopping early at epoch {epoch}: no improvement in "
                    f"{config.early_stopping_patience} epochs"
                )
                break

    if history.best_state is not None:
        model.load_state_dict(history.best_state)

    return history
