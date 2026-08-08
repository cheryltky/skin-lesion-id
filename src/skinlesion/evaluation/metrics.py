"""Evaluation metrics chosen for a long-tailed, clinically asymmetric problem.

Plain accuracy flatters a model on this dataset because melanocytic nevi
(ordinary moles) outnumber every malignant class combined. Macro-F1
weights every diagnosis equally, and top-k accuracy reflects a realistic
use of the tool: a differential-diagnosis shortlist, not a single forced
answer.

Neither of those is the metric that actually matters most here, though.
Mistaking one benign class for another is a false alarm; mistaking a
melanoma for a mole is a missed cancer. malignant_recall reports the one
number that speaks to that asymmetry directly: of the lesions that are
actually malignant or pre-malignant, what fraction did the model flag as
such -- regardless of which specific malignant subtype it guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    class_name: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ConfusionPair:
    true_class: str
    predicted_class: str
    count: int


@dataclass(frozen=True)
class MalignantRecall:
    """Recall on the malignant/pre-malignant super-class.

    A false negative here means the model called a dangerous lesion
    benign -- the failure mode that matters most clinically, independent
    of which specific malignant subtype was involved.
    """

    malignant_classes: tuple[str, ...]
    support: int
    missed: int
    recall: float


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    top_k_accuracy: float
    top_k: int
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    confusions: tuple[ConfusionPair, ...]
    malignant_recall: MalignantRecall | None

    def weakest_classes(self, limit: int = 5) -> tuple[ClassMetrics, ...]:
        return tuple(sorted(self.per_class, key=lambda m: m.f1)[:limit])


def _validate(probabilities: np.ndarray, labels: np.ndarray, num_classes: int) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-dimensional, got shape {probabilities.shape}")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probabilities has {probabilities.shape[0]} rows but there are "
            f"{labels.shape[0]} labels"
        )
    if probabilities.shape[1] != num_classes:
        raise ValueError(
            f"probabilities has {probabilities.shape[1]} columns but there are "
            f"{num_classes} classes"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels contain an index outside the class range")


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is in the k highest-scoring."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if labels.size == 0:
        return 0.0

    effective_k = min(k, probabilities.shape[1])
    # argpartition is O(n) versus a full sort, which matters on large test sets.
    top = np.argpartition(-probabilities, effective_k - 1, axis=1)[:, :effective_k]
    hits = (top == labels[:, None]).any(axis=1)
    return float(hits.mean())


def per_class_metrics(
    predictions: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> tuple[ClassMetrics, ...]:
    """Precision, recall and F1 for every class, including absent ones."""
    results: list[ClassMetrics] = []

    for index, name in enumerate(class_names):
        predicted_positive = predictions == index
        actually_positive = labels == index

        true_positives = int(np.sum(predicted_positive & actually_positive))
        false_positives = int(np.sum(predicted_positive & ~actually_positive))
        false_negatives = int(np.sum(~predicted_positive & actually_positive))

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        results.append(
            ClassMetrics(
                class_name=name,
                support=int(np.sum(actually_positive)),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return tuple(results)


def top_confusions(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    limit: int = 10,
) -> tuple[ConfusionPair, ...]:
    """The most frequent wrong-class pairs.

    Confusions between visually similar diagnoses (say, two benign
    keratoses) are a different kind of failure from a malignant lesion
    called benign, which is why malignant_recall exists separately -- this
    list alone would bury that distinction inside an aggregate count.
    """
    mistakes: dict[tuple[int, int], int] = {}
    for true_label, predicted in zip(labels, predictions):
        if true_label != predicted:
            key = (int(true_label), int(predicted))
            mistakes[key] = mistakes.get(key, 0) + 1

    ranked = sorted(mistakes.items(), key=lambda item: item[1], reverse=True)[:limit]
    return tuple(
        ConfusionPair(
            true_class=class_names[true_label],
            predicted_class=class_names[predicted],
            count=count,
        )
        for (true_label, predicted), count in ranked
    )


def malignant_recall(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    malignant_classes: Sequence[str],
) -> MalignantRecall | None:
    """Recall on the malignant/pre-malignant super-class.

    Collapses the multi-class problem into "malignant or pre-malignant"
    versus "benign" and reports recall on the former -- a missed melanoma
    counted as a miss regardless of whether the model guessed nevus or
    dermatofibroma instead. Returns None if no malignant class has any
    support in this split.
    """
    malignant_indices = {
        index for index, name in enumerate(class_names) if name in malignant_classes
    }
    if not malignant_indices:
        return None

    is_malignant_true = np.isin(labels, list(malignant_indices))
    is_malignant_pred = np.isin(predictions, list(malignant_indices))

    support = int(is_malignant_true.sum())
    if support == 0:
        return None

    caught = int(np.sum(is_malignant_true & is_malignant_pred))
    missed = support - caught

    return MalignantRecall(
        malignant_classes=tuple(malignant_classes),
        support=support,
        missed=missed,
        recall=caught / support,
    )


def evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 3,
    malignant_classes: Sequence[str] = (),
) -> EvaluationReport:
    """Build the full evaluation report from predicted probabilities."""
    _validate(probabilities, labels, len(class_names))

    predictions = probabilities.argmax(axis=1)
    per_class = per_class_metrics(predictions, labels, class_names)

    # Macro-F1 over classes that actually appear in this split.
    present = [m.f1 for m in per_class if m.support > 0]
    macro_f1 = float(np.mean(present)) if present else 0.0

    return EvaluationReport(
        accuracy=float((predictions == labels).mean()) if labels.size else 0.0,
        top_k_accuracy=top_k_accuracy(probabilities, labels, top_k),
        top_k=top_k,
        macro_f1=macro_f1,
        per_class=per_class,
        confusions=top_confusions(predictions, labels, class_names),
        malignant_recall=malignant_recall(predictions, labels, class_names, malignant_classes),
    )
