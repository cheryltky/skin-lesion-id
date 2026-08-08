"""Tests for the inference wrapper used by the app."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from skinlesion.config import ModelConfig
from skinlesion.modelling.model import CheckpointMetadata, build_model, save_checkpoint
from skinlesion.modelling.predict import Predictor

DIAGNOSES = ("nv", "mel", "bcc", "akiec")


@pytest.fixture
def checkpoint(tmp_path):
    config = ModelConfig(backbone="resnet18", pretrained=False, image_size=64, dropout=0.0)
    model = build_model(config, num_classes=len(DIAGNOSES))
    path = tmp_path / "model.pt"
    save_checkpoint(
        model,
        CheckpointMetadata(
            backbone="resnet18",
            image_size=64,
            class_names=DIAGNOSES,
            val_accuracy=0.77,
            temperature=1.25,
        ),
        path,
    )
    return path


@pytest.fixture
def image():
    return Image.new("RGB", (200, 150), (30, 140, 90))


def test_predictor_exposes_checkpoint_metadata(checkpoint):
    predictor = Predictor(checkpoint, device=torch.device("cpu"))

    assert predictor.diagnoses == DIAGNOSES
    assert predictor.metadata.val_accuracy == pytest.approx(0.77)
    assert predictor.metadata.temperature == pytest.approx(1.25)


def test_prediction_returns_top_k_candidates(checkpoint, image):
    predictor = Predictor(checkpoint, top_k=3, device=torch.device("cpu"))

    prediction = predictor.predict(image)

    assert len(prediction.candidates) == 3
    assert all(c.dx in DIAGNOSES for c in prediction.candidates)


def test_candidate_name_resolves_to_a_readable_diagnosis(checkpoint, image):
    predictor = Predictor(checkpoint, top_k=1, device=torch.device("cpu"))

    candidate = predictor.predict(image).best

    assert candidate.name != candidate.dx  # resolved via DIAGNOSIS_NAMES, not the raw code


def test_candidates_are_ordered_by_probability(checkpoint, image):
    predictor = Predictor(checkpoint, top_k=4, device=torch.device("cpu"))

    probabilities = [c.probability for c in predictor.predict(image).candidates]

    assert probabilities == sorted(probabilities, reverse=True)


def test_all_probabilities_form_a_distribution(checkpoint, image):
    predictor = Predictor(checkpoint, top_k=4, device=torch.device("cpu"))

    total = sum(c.probability for c in predictor.predict(image).candidates)

    assert total == pytest.approx(1.0, abs=1e-5)


def test_top_k_is_clamped_to_the_class_count(checkpoint, image):
    predictor = Predictor(checkpoint, top_k=99, device=torch.device("cpu"))

    assert len(predictor.predict(image).candidates) == len(DIAGNOSES)


def test_high_threshold_forces_abstention(checkpoint, image):
    predictor = Predictor(checkpoint, abstain_threshold=1.0, device=torch.device("cpu"))

    prediction = predictor.predict(image)

    assert prediction.abstained
    assert "Not confident enough" in prediction.summary


def test_zero_threshold_never_abstains(checkpoint, image):
    predictor = Predictor(checkpoint, abstain_threshold=0.0, device=torch.device("cpu"))

    prediction = predictor.predict(image)

    assert not prediction.abstained
    assert "Most likely" in prediction.summary


def test_greyscale_images_are_accepted(checkpoint):
    predictor = Predictor(checkpoint, device=torch.device("cpu"))

    prediction = predictor.predict(Image.new("L", (120, 120), 128))

    assert len(prediction.candidates) >= 1


def test_percentage_matches_probability(checkpoint, image):
    predictor = Predictor(checkpoint, device=torch.device("cpu"))

    candidate = predictor.predict(image).best

    assert candidate.percentage == pytest.approx(candidate.probability * 100.0)


def test_invalid_threshold_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="abstain_threshold"):
        Predictor(checkpoint, abstain_threshold=1.5, device=torch.device("cpu"))


def test_invalid_top_k_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="top_k"):
        Predictor(checkpoint, top_k=0, device=torch.device("cpu"))


def test_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        Predictor(tmp_path / "absent.pt", device=torch.device("cpu"))
