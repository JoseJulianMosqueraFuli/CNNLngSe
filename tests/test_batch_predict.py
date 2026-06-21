"""
Tests para predicción por lotes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sign_classifier.batch_predict import (
    discover_images,
    predict_batch,
    save_predictions_csv,
)
from sign_classifier.model import create_model


def _create_image(path: Path, size: tuple = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    img.save(path)


@pytest.fixture
def sample_image_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        _create_image(img_dir / "img1.jpg")
        _create_image(img_dir / "img2.png")
        _create_image(img_dir / "sub" / "img3.jpg")
        yield str(img_dir)


def test_discover_images(sample_image_dir):
    images = discover_images(sample_image_dir)
    assert len(images) == 3


def test_discover_images_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        images = discover_images(tmpdir)
        assert images == []


def test_discover_images_missing_dir():
    with pytest.raises(FileNotFoundError):
        discover_images("/ruta/inexistente")


def test_predict_batch(sample_image_dir):
    model = create_model((64, 64, 3), 3)
    classes = ["a", "b", "c"]
    image_paths = discover_images(sample_image_dir)

    results = predict_batch(model, image_paths, classes)

    assert len(results) == 3
    for row in results:
        assert "image_path" in row
        assert "predicted_class" in row
        assert "confidence" in row
        assert "prob_a" in row
        assert "prob_b" in row
        assert "prob_c" in row


def test_save_predictions_csv(tmp_path):
    results = [
        {
            "image_path": "a.jpg",
            "predicted_class": "a",
            "confidence": 0.9,
            "prob_a": 0.9,
            "prob_b": 0.1,
        }
    ]
    output = tmp_path / "out.csv"
    save_predictions_csv(results, str(output))

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "image_path,predicted_class,confidence,prob_a,prob_b" in content
    assert "a.jpg,a,0.9,0.9,0.1" in content
