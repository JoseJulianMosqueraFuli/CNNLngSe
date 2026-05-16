"""
Módulo de entrenamiento para el clasificador de señas.

Implementa el flujo de entrenamiento del modelo CNN usando APIs modernas
de TensorFlow/Keras.
"""

import logging
from pathlib import Path

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_SHAPE,
    EPOCHS,
    BATCH_SIZE,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    MODEL_PATH,
    LOG_DIR,
    NUM_CLASSES,
)
from .model import create_model
from .data_loader import create_data_generators

logger = logging.getLogger(__name__)


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_path: str = TRAINING_DATA_PATH,
    val_path: str = VALIDATION_DATA_PATH,
    model_path: str = MODEL_PATH,
    verbose: int = 1
):
    """
    Entrena el modelo CNN con los datos proporcionados.

    Args:
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        train_path: Ruta a los datos de entrenamiento
        val_path: Ruta a los datos de validación
        model_path: Ruta donde guardar el modelo entrenado
        verbose: Nivel de verbosidad (0, 1, 2)

    Returns:
        Tupla (modelo entrenado, historial de entrenamiento)
    """
    # Crear directorio para el modelo si no existe
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Crear generadores de datos
    train_generator, val_generator = create_data_generators(
        train_path=train_path,
        val_path=val_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch_size
    )

    # Crear modelo
    model = create_model(
        input_shape=IMAGE_SHAPE,
        num_classes=NUM_CLASSES
    )

    # Configurar callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=verbose
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=verbose
        ),
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    # Entrenar modelo usando fit() (no fit_generator() deprecado)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=verbose
    )

    # Guardar modelo final en formato .keras
    model.save(model_path)

    return model, history


def main():
    """Punto de entrada principal para entrenamiento."""
    logger.info("Iniciando entrenamiento del clasificador de señas...")
    logger.info("Épocas: %d", EPOCHS)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Datos de entrenamiento: %s", TRAINING_DATA_PATH)
    logger.info("Datos de validación: %s", VALIDATION_DATA_PATH)
    logger.info("Modelo se guardará en: %s", MODEL_PATH)

    _, history = train_model()

    best_val_acc = max(history.history['val_accuracy'])
    logger.info("Entrenamiento completado. Mejor accuracy: %.4f", best_val_acc)
    logger.info("Modelo guardado en: %s", MODEL_PATH)


if __name__ == "__main__":
    main()
