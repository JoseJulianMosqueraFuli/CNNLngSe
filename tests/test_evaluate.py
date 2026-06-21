"""
Tests para el módulo de evaluación.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sign_classifier.config import IMAGE_HEIGHT, IMAGE_WIDTH
from sign_classifier.evaluate import evaluate_model
from sign_classifier.exceptions import ModelError
from sign_classifier.model import create_model


def _create_image(path: Path, size: tuple = (IMAGE_HEIGHT, IMAGE_WIDTH)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    img.save(path)


@pytest.fixture
def sample_val_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        val_dir = Path(tmpdir) / "val"
        for cls in ["a", "b"]:
            _create_image(val_dir / cls / f"{cls}_1.jpg")
        yield str(val_dir)


def test_evaluate_model_returns_metrics(sample_val_dataset):
    model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 2)
    metrics = evaluate_model(model, sample_val_dataset)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics


def test_evaluate_model_none_model():
    with pytest.raises(ModelError, match="Se requiere un modelo"):
        evaluate_model(None, "/tmp/fake")
