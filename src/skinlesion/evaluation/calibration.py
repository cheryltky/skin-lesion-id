"""Confidence calibration.

A fine-tuned network is usually overconfident: it will report 0.95 on
predictions that are right about 80 percent of the time. That matters here
because the app shows a confidence figure to someone deciding whether a
lesion is worth a clinician's attention, and because the abstain threshold
is meaningless if the underlying numbers are inflated.

Temperature scaling fits a single scalar on the validation set to correct
this. It cannot change which class ranks highest, so accuracy is untouched
while the reported confidence becomes honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed gap. Positive means overconfident."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    temperature: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def is_overconfident(self) -> bool:
        weighted = sum(b.gap * b.count for b in self.bins)
        total = sum(b.count for b in self.bins)
        return total > 0 and weighted / total > 0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with an optional temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> tuple[ReliabilityBin, ...]:
    """Bucket predictions by confidence and compare against actual accuracy."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must describe the same samples")

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ReliabilityBin] = []

    for index in range(num_bins):
        lower, upper = edges[index], edges[index + 1]
        # Include the right edge in the final bin so confidence 1.0 counts.
        if index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_confidence=float(confidences[mask].mean()) if count else 0.0,
                accuracy=float(correct[mask].mean()) if count else 0.0,
            )
        )

    return tuple(bins)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> float:
    """Mean absolute gap between confidence and accuracy, weighted by count.

    Zero is perfect. Above roughly 0.05 is worth reporting as a caveat.
    """
    bins = reliability_bins(probabilities, labels, num_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    return float(sum(abs(b.gap) * b.count for b in bins) / total)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iterations: int = 200,
    learning_rate: float = 0.01,
) -> float:
    """Fit the temperature that minimises validation negative log likelihood.

    Must be fitted on validation data, never on test, or the reported
    calibration is measuring itself.
    """
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must describe the same samples")
    if logits.shape[0] == 0:
        raise ValueError("cannot fit a temperature on zero samples")

    logit_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    # Optimise log-temperature so temperature stays strictly positive.
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=learning_rate, max_iter=max_iterations)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = criterion(logit_tensor / log_temperature.exp(), label_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)

    return float(log_temperature.detach().exp().item())


def calibrate(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    num_bins: int = 15,
) -> tuple[CalibrationReport, CalibrationReport]:
    """Report calibration before and after temperature scaling.

    Temperature is fitted on validation and both reports are computed on
    test, which is the honest way to show the correction generalises.
    """
    temperature = fit_temperature(validation_logits, validation_labels)

    before = softmax(test_logits, temperature=1.0)
    after = softmax(test_logits, temperature=temperature)

    uncalibrated = CalibrationReport(
        expected_calibration_error=expected_calibration_error(before, test_labels, num_bins),
        temperature=1.0,
        bins=reliability_bins(before, test_labels, num_bins),
    )
    calibrated = CalibrationReport(
        expected_calibration_error=expected_calibration_error(after, test_labels, num_bins),
        temperature=temperature,
        bins=reliability_bins(after, test_labels, num_bins),
    )

    return uncalibrated, calibrated
