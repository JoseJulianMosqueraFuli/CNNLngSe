import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

from .exceptions import DataLoadingError

logger = logging.getLogger(__name__)

AUGMENTATION = tf.keras.Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(factor=20 / 360.0),
        RandomZoom(height_factor=0.1, width_factor=0.1),
        RandomBrightness(factor=0.2),
        RandomContrast(factor=0.2),
    ]
)


def _parse_class_names(path: str) -> list[str]:
    data_dir = Path(path)
    if not data_dir.exists():
        raise DataLoadingError(f"Directorio no encontrado: {path}")
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise DataLoadingError(f"No se encontraron subdirectorios (clases) en: {path}")
    return class_names


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = AUGMENTATION(image, training=True)
    return image, label


def _normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_validation_dataset(
    val_path: str,
    target_size: tuple,
    batch_size: int,
) -> tuple:
    """
    Crea el dataset de validación a partir de un directorio de imágenes.

    Args:
        val_path: Ruta al directorio de validación.
        target_size: Tamaño al que redimensionar las imágenes (height, width).
        batch_size: Tamaño del batch.

    Returns:
        Tupla (dataset de validación, nombres de clases).
    """
    class_names = _parse_class_names(val_path)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    val_ds = val_ds.map(_normalize, num_parallel_calls=autotune)
    val_ds = val_ds.prefetch(autotune)

    val_size = sum(1 for _ in Path(val_path).rglob("*") if _.is_file())
    if val_size == 0:
        raise DataLoadingError(f"No se encontraron imágenes en validación: {val_path}")

    logger.info(
        "Datos de validación cargados: %d imágenes, %d clases",
        val_size,
        len(class_names),
    )

    return val_ds, class_names


def create_data_generators(
    train_path: str,
    val_path: str,
    target_size: tuple,
    batch_size: int,
) -> tuple:
    """
    Crea los datasets de entrenamiento y validación.

    Args:
        train_path: Ruta al directorio de entrenamiento.
        val_path: Ruta al directorio de validación.
        target_size: Tamaño al que redimensionar las imágenes (height, width).
        batch_size: Tamaño del batch.

    Returns:
        Tupla (dataset de entrenamiento, dataset de validación, nombres de clases).
    """
    class_names = _parse_class_names(train_path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds, val_class_names = create_validation_dataset(
        val_path=val_path,
        target_size=target_size,
        batch_size=batch_size,
    )

    if sorted(class_names) != sorted(val_class_names):
        raise DataLoadingError(
            f"Clases de entrenamiento ({class_names}) no coinciden "
            f"con clases de validación ({val_class_names})"
        )

    autotune = tf.data.AUTOTUNE
    # Aplicar augmentación antes de la normalización para que las capas Keras
    # trabajen en el rango original de la imagen [0, 255].
    train_ds = train_ds.map(_augment, num_parallel_calls=autotune)
    train_ds = train_ds.map(_normalize, num_parallel_calls=autotune)
    train_ds = train_ds.prefetch(autotune)

    train_size = sum(1 for _ in Path(train_path).rglob("*") if _.is_file())
    if train_size == 0:
        raise DataLoadingError(
            f"No se encontraron imágenes en entrenamiento: {train_path}"
        )

    logger.info(
        "Datos cargados: %d entrenamiento, %d validación, %d clases",
        train_size,
        sum(1 for _ in Path(val_path).rglob("*") if _.is_file()),
        len(class_names),
    )

    return train_ds, val_ds, class_names
