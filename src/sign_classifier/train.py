"""
Módulo de entrenamiento para el clasificador de señas.

Implementa el flujo de entrenamiento del modelo CNN usando APIs modernas
de TensorFlow/Keras.
"""

import logging
from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from .config import (
    BATCH_SIZE,
    CLASSES,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_SHAPE,
    IMAGE_WIDTH,
    LOG_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from .data_loader import create_data_generators
from .exceptions import ConfigurationError
from .model import create_model

logger = logging.getLogger(__name__)


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_path: str = TRAINING_DATA_PATH,
    val_path: str = VALIDATION_DATA_PATH,
    model_path: str = MODEL_PATH,
    verbose: int = 1,
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

    # Crear datasets de datos
    train_ds, val_ds, class_names = create_data_generators(
        train_path=train_path,
        val_path=val_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch_size,
    )

    # Validar que las clases del dataset coincidan con la configuración
    if sorted(class_names) != sorted(CLASSES):
        raise ConfigurationError(
            f"Clases en datos ({class_names}) no coinciden con config ({CLASSES})"
        )
    if len(class_names) != NUM_CLASSES:
        raise ConfigurationError(
            f"Número de clases en datos ({len(class_names)}) "
            f"no coincide con NUM_CLASSES ({NUM_CLASSES})"
        )

    # Crear modelo
    model = create_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)

    # Configurar callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=verbose,
        ),
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=verbose
        ),
        TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True, update_freq="epoch"
        ),
    ]

    # Entrenar modelo
    # ModelCheckpoint ya guarda el mejor modelo en model_path, por lo que no es
    # necesario volver a guardar al final (evita sobrescribir el mejor con el
    # último estado).
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose,
    )

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

    best_val_acc = max(history.history["val_accuracy"])
    logger.info("Entrenamiento completado. Mejor accuracy: %.4f", best_val_acc)
    logger.info("Modelo guardado en: %s", MODEL_PATH)


if __name__ == "__main__":
    main()
