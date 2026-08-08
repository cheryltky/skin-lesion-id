"""Tests for confidence calibration."""

from __future__ import annotations

import numpy as np
import pytest

from skinlesion.evaluation.calibration import (
    calibrate,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    softmax,
)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))

    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_softmax_is_stable_for_large_logits():
    probabilities = softmax(np.array([[1000.0, 1001.0]]))

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_higher_temperature_softens_the_distribution():
    logits = np.array([[3.0, 1.0, 0.0]])

    sharp = softmax(logits, temperature=1.0).max()
    soft = softmax(logits, temperature=5.0).max()

    assert soft < sharp


def test_temperature_scaling_preserves_the_ranking():
    logits = np.array([[3.0, 1.0, 0.0], [0.5, 2.5, 1.0]])

    cold = softmax(logits, temperature=0.5).argmax(axis=1)
    hot = softmax(logits, temperature=4.0).argmax(axis=1)

    assert (cold == hot).all()


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        softmax(np.array([[1.0, 2.0]]), temperature=0.0)


def test_perfectly_calibrated_predictions_have_near_zero_error():
    # Confidence 1.0 and always correct.
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])

    assert expected_calibration_error(probabilities, labels, num_bins=5) == pytest.approx(0.0)


def test_overconfident_predictions_produce_a_large_error():
    # Claims 0.99 confidence but is wrong half the time.
    probabilities = np.array([[0.99, 0.01]] * 4)
    labels = np.array([0, 1, 0, 1])

    error = expected_calibration_error(probabilities, labels, num_bins=10)

    assert error == pytest.approx(0.49, abs=0.02)


def test_reliability_bins_cover_the_unit_interval():
    probabilities = np.array([[0.6, 0.4], [1.0, 0.0]])
    labels = np.array([0, 0])

    bins = reliability_bins(probabilities, labels, num_bins=10)

    assert len(bins) == 10
    assert bins[0].lower == pytest.approx(0.0)
    assert bins[-1].upper == pytest.approx(1.0)
    # Confidence of exactly 1.0 must fall in the final bin, not be dropped.
    assert sum(b.count for b in bins) == 2


def test_reliability_bins_reject_too_few_bins():
    with pytest.raises(ValueError, match="num_bins must be at least 2"):
        reliability_bins(np.array([[0.5, 0.5]]), np.array([0]), num_bins=1)


def test_fit_temperature_exceeds_one_for_overconfident_logits():
    rng = np.random.default_rng(3)
    # Large logit gaps, but the argmax is only right about 70 percent of
    # the time, so the fitted temperature should soften the outputs.
    logits = rng.normal(scale=1.0, size=(400, 4))
    logits[:, 0] += 6.0
    labels = np.where(rng.random(400) < 0.7, 0, rng.integers(1, 4, size=400))

    assert fit_temperature(logits, labels) > 1.0


def test_fit_temperature_rejects_empty_input():
    with pytest.raises(ValueError, match="zero samples"):
        fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_calibration_reduces_error_on_overconfident_logits():
    rng = np.random.default_rng(11)
    logits = rng.normal(scale=1.0, size=(600, 5))
    logits[:, 0] += 7.0
    labels = np.where(rng.random(600) < 0.65, 0, rng.integers(1, 5, size=600))

    split = 300
    before, after = calibrate(
        logits[:split], labels[:split], logits[split:], labels[split:]
    )

    assert before.temperature == pytest.approx(1.0)
    assert after.temperature > 1.0
    assert after.expected_calibration_error < before.expected_calibration_error
    assert before.is_overconfident
