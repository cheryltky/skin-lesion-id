"""Inference wrapper used by the Streamlit app.

Confidence is temperature-scaled using the value fitted during training,
and a prediction below the abstain threshold is reported as uncertain
rather than dressed up as a diagnosis. This is a portfolio project, not a
medical device: every prediction carries that disclaimer, and the tool
declines to answer rather than let a confident-looking number substitute
for a clinician.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..data.datasets import build_eval_transform
from ..data.metadata import DIAGNOSIS_NAMES
from .model import CheckpointMetadata, load_checkpoint, select_device

NOT_MEDICAL_ADVICE = (
    "This is a portfolio machine learning project, not a diagnostic device. "
    "It has not been clinically validated and must never substitute for "
    "assessment by a qualified clinician. If you are concerned about a "
    "lesion on your own skin, see a doctor."
)


@dataclass(frozen=True)
class Candidate:
    dx: str
    probability: float

    @property
    def name(self) -> str:
        return DIAGNOSIS_NAMES.get(self.dx, self.dx)

    @property
    def percentage(self) -> float:
        return self.probability * 100.0


@dataclass(frozen=True)
class Prediction:
    candidates: tuple[Candidate, ...]
    abstained: bool
    temperature: float

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    @property
    def summary(self) -> str:
        if self.abstained:
            return (
                "Not confident enough to suggest a diagnosis for this image. "
                "The closest matches are listed below, but treat them as "
                "unreliable suggestions, not a shortlist."
            )
        return f"Most likely {self.best.name} at {self.best.percentage:.1f} percent confidence."


class Predictor:
    """Loads a checkpoint once and classifies images against it."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        abstain_threshold: float = 0.5,
        top_k: int = 3,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must lie in [0, 1]")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self._device = device or select_device()
        self._model, self._metadata = load_checkpoint(checkpoint_path, self._device)
        self._transform = build_eval_transform(self._metadata.image_size)
        self._abstain_threshold = abstain_threshold
        self._top_k = min(top_k, self._metadata.num_classes)

    @property
    def metadata(self) -> CheckpointMetadata:
        return self._metadata

    @property
    def diagnoses(self) -> tuple[str, ...]:
        return self._metadata.class_names

    def predict(self, image: Image.Image) -> Prediction:
        """Classify one image, returning calibrated top-k candidates."""
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            # Apply the temperature fitted on validation data before softmax.
            scaled = logits / self._metadata.temperature
            probabilities = torch.softmax(scaled, dim=1).squeeze(0).cpu().numpy()

        order = np.argsort(-probabilities)[: self._top_k]
        candidates = tuple(
            Candidate(dx=self._metadata.class_names[i], probability=float(probabilities[i]))
            for i in order
        )

        return Prediction(
            candidates=candidates,
            abstained=candidates[0].probability < self._abstain_threshold,
            temperature=self._metadata.temperature,
        )
