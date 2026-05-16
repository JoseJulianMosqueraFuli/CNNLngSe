import logging
from pathlib import Path

import tensorflow as tf

from .config import IMAGE_HEIGHT, IMAGE_WIDTH

logger = logging.getLogger(__name__)

AUGMENTATION = {
    "flip": True,
    "brightness_max_delta": 0.2,
    "contrast_lower": 0.8,
    "contrast_upper": 1.2,
    "rotation_max_degrees": 20,
}


def _parse_class_names(path: str) -> list[str]:
    data_dir = Path(path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {path}")
    class_names = sorted(
        [d.name for d in data_dir.iterdir() if d.is_dir()]
    )
    if not class_names:
        raise ValueError(
            f"No se encontraron subdirectorios (clases) en: {path}"
        )
    return class_names


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    if AUGMENTATION["flip"]:
        image = tf.image.random_flip_left_right(image)
    if AUGMENTATION["brightness_max_delta"]:
        image = tf.image.random_brightness(
            image, max_delta=AUGMENTATION["brightness_max_delta"]
        )
    if AUGMENTATION["contrast_lower"] and AUGMENTATION["contrast_upper"]:
        image = tf.image.random_contrast(
            image, lower=AUGMENTATION["contrast_lower"],
            upper=AUGMENTATION["contrast_upper"]
        )
    if AUGMENTATION["rotation_max_degrees"]:
        angle = tf.random.uniform(
            [], -AUGMENTATION["rotation_max_degrees"],
            AUGMENTATION["rotation_max_degrees"]
        ) * (3.14159265 / 180.0)
        image = tfa_image_rotate(image, angle)
    return image, label


def tfa_image_rotate(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    original_dtype = image.dtype
    image = tf.expand_dims(image, 0)
    sin = tf.sin(angle)
    cos = tf.cos(angle)
    transform = [cos, -sin, 0, sin, cos, 0, 0, 0]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=[transform],
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    return tf.cast(tf.squeeze(image, 0), original_dtype)


def _normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_data_generators(
    train_path: str,
    val_path: str,
    target_size: tuple,
    batch_size: int,
) -> tuple:
    class_names = _parse_class_names(train_path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(_normalize, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.map(_normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    train_size = sum(1 for _ in Path(train_path).rglob("*") if _.is_file())
    val_size = sum(1 for _ in Path(val_path).rglob("*") if _.is_file())

    if train_size == 0:
        raise ValueError(
            f"No se encontraron imágenes en entrenamiento: {train_path}"
        )
    if val_size == 0:
        raise ValueError(
            f"No se encontraron imágenes en validación: {val_path}"
        )

    logger.info(
        "Datos cargados: %d entrenamiento, %d validación, %d clases",
        train_size, val_size, len(class_names)
    )

    return train_ds, val_ds, class_names
