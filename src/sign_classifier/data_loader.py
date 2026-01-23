"""
Módulo de carga de datos para el clasificador de señas.

Maneja la preparación y augmentación de datos usando ImageDataGenerator.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    train_path: str,
    val_path: str,
    target_size: tuple,
    batch_size: int
) -> tuple:
    """
    Crea generadores de datos para entrenamiento y validación.

    Args:
        train_path: Ruta al directorio de datos de entrenamiento
        val_path: Ruta al directorio de datos de validación
        target_size: Tupla (height, width) para redimensionar imágenes
        batch_size: Tamaño del batch para el generador

    Returns:
        Tupla (train_generator, validation_generator)

    Raises:
        ValueError: Si los directorios están vacíos o no contienen
            imágenes válidas
    """
    # Generador de entrenamiento con augmentación de datos
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generador de validación solo con normalización
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Crear generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Crear generador de validación
    validation_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Verificar que hay datos
    if train_generator.samples == 0:
        raise ValueError(
            f"No se encontraron imágenes en entrenamiento: {train_path}"
        )

    if validation_generator.samples == 0:
        raise ValueError(
            f"No se encontraron imágenes en validación: {val_path}"
        )

    return train_generator, validation_generator
