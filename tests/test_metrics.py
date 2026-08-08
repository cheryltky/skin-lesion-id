"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from skinlesion.evaluation.metrics import (
    evaluate,
    malignant_recall,
    per_class_metrics,
    top_confusions,
    top_k_accuracy,
)

CLASSES = ("nv", "bkl", "mel")


def test_top_1_accuracy_matches_argmax():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    labels = np.array([0, 1, 0])

    assert top_k_accuracy(probabilities, labels, k=1) == pytest.approx(2 / 3)


def test_top_k_is_monotonic_in_k():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(3), size=50)
    labels = rng.integers(0, 3, size=50)

    scores = [top_k_accuracy(probabilities, labels, k) for k in (1, 2, 3)]

    assert scores[0] <= scores[1] <= scores[2]
    assert scores[2] == pytest.approx(1.0)


def test_top_k_larger_than_class_count_is_clamped():
    probabilities = np.array([[0.6, 0.4]])
    labels = np.array([1])

    assert top_k_accuracy(probabilities, labels, k=99) == pytest.approx(1.0)


def test_top_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        top_k_accuracy(np.array([[0.5, 0.5]]), np.array([0]), k=0)


def test_perfect_predictions_give_f1_of_one():
    predictions = np.array([0, 1, 2])
    labels = np.array([0, 1, 2])

    for metric in per_class_metrics(predictions, labels, CLASSES):
        assert metric.f1 == pytest.approx(1.0)
        assert metric.support == 1


def test_class_never_predicted_has_zero_recall():
    predictions = np.array([0, 0, 0])
    labels = np.array([0, 1, 1])

    by_name = {m.class_name: m for m in per_class_metrics(predictions, labels, CLASSES)}

    assert by_name["bkl"].recall == pytest.approx(0.0)
    assert by_name["bkl"].support == 2
    # Absent from both predictions and labels, so precision and recall are
    # defined as zero rather than raising.
    assert by_name["mel"].f1 == pytest.approx(0.0)


def test_macro_f1_ignores_classes_with_no_support():
    # Only classes 0 and 1 appear; class 2 should not drag the mean down.
    probabilities = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    labels = np.array([0, 1])

    report = evaluate(probabilities, labels, CLASSES, top_k=2)

    assert report.macro_f1 == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(1.0)


def test_confusions_are_ranked_by_frequency():
    predictions = np.array([1, 1, 1, 2])
    labels = np.array([0, 0, 0, 0])

    confusions = top_confusions(predictions, labels, CLASSES)

    assert confusions[0].true_class == "nv"
    assert confusions[0].predicted_class == "bkl"
    assert confusions[0].count == 3
    assert confusions[1].count == 1


def test_correct_predictions_are_not_counted_as_confusions():
    predictions = np.array([0, 1, 2])
    labels = np.array([0, 1, 2])

    assert top_confusions(predictions, labels, CLASSES) == ()


def test_weakest_classes_are_returned_worst_first():
    probabilities = np.array(
        [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]
    )
    labels = np.array([0, 1, 2])

    weakest = evaluate(probabilities, labels, CLASSES, top_k=2).weakest_classes(limit=1)

    assert weakest[0].class_name == "bkl"


def test_shape_mismatches_are_rejected():
    with pytest.raises(ValueError, match="rows but there are"):
        evaluate(np.zeros((3, 3)), np.array([0, 1]), CLASSES)

    with pytest.raises(ValueError, match="columns but there are"):
        evaluate(np.zeros((2, 5)), np.array([0, 1]), CLASSES)


def test_out_of_range_labels_are_rejected():
    with pytest.raises(ValueError, match="outside the class range"):
        evaluate(np.zeros((2, 3)), np.array([0, 9]), CLASSES)


# --- malignant_recall: the clinically load-bearing metric ---------------


def test_malignant_recall_counts_missed_melanomas_as_misses_regardless_of_guess():
    # Two melanomas: one correctly caught, one called "nv" (benign) --
    # a miss -- and one called "bkl" (also benign, wrong subtype AND
    # wrong super-class) -- also a miss. Recall should be 1/3.
    predictions = np.array([2, 0, 1])
    labels = np.array([2, 2, 2])

    result = malignant_recall(predictions, labels, CLASSES, malignant_classes=("mel",))

    assert result is not None
    assert result.support == 3
    assert result.missed == 2
    assert result.recall == pytest.approx(1 / 3)


def test_malignant_recall_is_indifferent_to_which_malignant_subtype_is_guessed():
    classes = ("nv", "mel", "bcc")
    predictions = np.array([2, 1])  # guessed bcc, guessed mel
    labels = np.array([1, 2])  # actually mel, actually bcc

    result = malignant_recall(
        predictions, labels, classes, malignant_classes=("mel", "bcc")
    )

    # Both wrong about the specific subtype, but both correctly flagged as
    # "some malignant thing" -- full recall on the super-class.
    assert result.recall == pytest.approx(1.0)
    assert result.missed == 0


def test_malignant_recall_returns_none_when_no_malignant_class_is_configured():
    predictions = np.array([0, 1])
    labels = np.array([0, 1])

    assert malignant_recall(predictions, labels, CLASSES, malignant_classes=()) is None


def test_malignant_recall_returns_none_when_malignant_class_has_no_support():
    predictions = np.array([0, 1])
    labels = np.array([0, 1])

    # "mel" is a configured malignant class but never appears as a true label.
    assert malignant_recall(predictions, labels, CLASSES, malignant_classes=("mel",)) is None


def test_evaluate_includes_malignant_recall_when_configured():
    probabilities = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    labels = np.array([2, 0])

    report = evaluate(probabilities, labels, CLASSES, top_k=2, malignant_classes=("mel",))

    assert report.malignant_recall is not None
    assert report.malignant_recall.recall == pytest.approx(1.0)
