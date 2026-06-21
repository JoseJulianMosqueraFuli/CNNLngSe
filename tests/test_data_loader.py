"""
Tests para el módulo de carga de datos.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sign_classifier.data_loader import (
    create_data_generators,
    create_validation_dataset,
)
from sign_classifier.exceptions import DataLoadingError


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

        for cls in ["a", "b"]:
            _create_image(train_dir / cls / f"{cls}_1.jpg")
            _create_image(train_dir / cls / f"{cls}_2.jpg")
            _create_image(val_dir / cls / f"{cls}_1.jpg")

        yield str(train_dir), str(val_dir)


def test_create_data_generators_returns_datasets(sample_dataset):
    train_path, val_path = sample_dataset
    train_ds, val_ds, class_names = create_data_generators(
        train_path=train_path,
        val_path=val_path,
        target_size=(64, 64),
        batch_size=2,
    )

    assert class_names == ["a", "b"]
    assert train_ds is not None
    assert val_ds is not None


def test_create_validation_dataset_returns_dataset(sample_dataset):
    _, val_path = sample_dataset
    val_ds, class_names = create_validation_dataset(
        val_path=val_path,
        target_size=(64, 64),
        batch_size=2,
    )

    assert class_names == ["a", "b"]
    assert val_ds is not None


def test_create_data_generators_missing_train_dir():
    with pytest.raises(DataLoadingError, match="Directorio no encontrado"):
        create_data_generators(
            train_path="/ruta/inexistente",
            val_path="/otra/ruta/inexistente",
            target_size=(64, 64),
            batch_size=2,
        )


def test_create_data_generators_empty_classes():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(DataLoadingError, match="No se encontraron subdirectorios"),
    ):
        create_data_generators(
            train_path=tmpdir,
            val_path=tmpdir,
            target_size=(64, 64),
            batch_size=2,
        )


def test_create_data_generators_mismatched_classes():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        val_dir = Path(tmpdir) / "val"
        _create_image(train_dir / "a" / "a.jpg")
        _create_image(val_dir / "b" / "b.jpg")

        with pytest.raises(DataLoadingError, match="no coinciden"):
            create_data_generators(
                train_path=str(train_dir),
                val_path=str(val_dir),
                target_size=(64, 64),
                batch_size=2,
            )
