"""
Tests para el módulo de entrenamiento.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sign_classifier.config import CLASSES
from sign_classifier.exceptions import ConfigurationError
from sign_classifier.train import train_model


def _create_image(path: Path, size: tuple = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    img.save(path)


@pytest.fixture
def sample_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        val_dir = Path(tmpdir) / "val"
        model_path = Path(tmpdir) / "modelo" / "modelo.keras"

        for cls in CLASSES:
            _create_image(train_dir / cls / f"{cls}_1.jpg")
            _create_image(train_dir / cls / f"{cls}_2.jpg")
            _create_image(val_dir / cls / f"{cls}_1.jpg")

        yield str(train_dir), str(val_dir), str(model_path)


def test_train_model_runs(sample_dataset):
    train_path, val_path, model_path = sample_dataset
    model, history = train_model(
        epochs=1,
        batch_size=2,
        train_path=train_path,
        val_path=val_path,
        model_path=model_path,
        verbose=0,
    )

    assert model is not None
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert Path(model_path).exists()


def test_train_model_invalid_classes():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        val_dir = Path(tmpdir) / "val"
        model_path = Path(tmpdir) / "modelo" / "modelo.keras"

        # Crear clases que no coinciden con CLASSES
        for cls in ["x", "y", "z"]:
            _create_image(train_dir / cls / f"{cls}_1.jpg")
            _create_image(val_dir / cls / f"{cls}_1.jpg")

        with pytest.raises(ConfigurationError, match="no coinciden"):
            train_model(
                epochs=1,
                batch_size=2,
                train_path=str(train_dir),
                val_path=str(val_dir),
                model_path=str(model_path),
                verbose=0,
            )
