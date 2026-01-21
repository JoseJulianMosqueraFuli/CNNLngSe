"""
Módulo del modelo CNN para clasificación de señas.

Define la arquitectura de la red neuronal convolucional mejorada.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from .config import LEARNING_RATE


def create_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Crea y retorna el modelo CNN compilado.

    Args:
        input_shape: Tupla (height, width, channels)
        num_classes: Número de clases a clasificar

    Returns:
        Modelo Keras compilado

    Raises:
        ValueError: Si input_shape o num_classes son inválidos
    """
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError(
            "input_shape debe ser una tupla de 3 elementos (height, width, channels)"
        )
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes debe ser un entero >= 2")

    model = Sequential([
        # Bloque Convolucional 1 (32 filtros)
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque Convolucional 2 (64 filtros)
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque Convolucional 3 (128 filtros)
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Capas Densas
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
