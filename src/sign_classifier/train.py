"""
Módulo de entrenamiento para el clasificador de señas.

Implementa el flujo de entrenamiento del modelo CNN usando APIs modernas
de TensorFlow/Keras.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from tensorflow.keras.callbacks import (
    CSVLogger,
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
    METRICS_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    TRAINING_DATA_PATH,
    TRANSFER_LEARNING_BACKBONE,
    USE_TRANSFER_LEARNING,
    VALIDATION_DATA_PATH,
    setup_logging,
)
from .data_loader import create_data_generators
from .exceptions import ConfigurationError
from .model import create_model, create_transfer_learning_model

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
    # Crear directorios para el modelo y métricas si no existen
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

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
    if USE_TRANSFER_LEARNING:
        logger.info(
            "Usando transfer learning con backbone: %s",
            TRANSFER_LEARNING_BACKBONE,
        )
        model = create_transfer_learning_model(
            input_shape=IMAGE_SHAPE,
            num_classes=NUM_CLASSES,
            backbone=TRANSFER_LEARNING_BACKBONE,
        )
    else:
        model = create_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)

    # Configurar callbacks
    csv_path = metrics_dir / "history.csv"
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
        CSVLogger(filename=str(csv_path), append=False, separator=","),
    ]

    # TensorBoard es opcional; si no está instalado, se omite el callback.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        logger.warning(
            "TensorBoard no está instalado. El entrenamiento continuará sin "
            "registro de TensorBoard."
        )
    else:
        callbacks.append(
            TensorBoard(
                log_dir=LOG_DIR, histogram_freq=1, write_graph=True, update_freq="epoch"
            )
        )

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


def _save_metrics_summary(history, metrics_dir: Path) -> None:
    """Guarda un resumen de métricas de entrenamiento en JSON."""
    history_dict = history.history
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "epochs_trained": len(history_dict.get("loss", [])),
        "best_train_accuracy": max(history_dict.get("accuracy", [0.0])),
        "best_val_accuracy": max(history_dict.get("val_accuracy", [0.0])),
        "best_train_loss": min(history_dict.get("loss", [float("inf")])),
        "best_val_loss": min(history_dict.get("val_loss", [float("inf")])),
        "final_train_accuracy": history_dict.get("accuracy", [0.0])[-1],
        "final_val_accuracy": history_dict.get("val_accuracy", [0.0])[-1],
        "final_train_loss": history_dict.get("loss", [float("inf")])[-1],
        "final_val_loss": history_dict.get("val_loss", [float("inf")])[-1],
    }
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Métricas guardadas en: %s", metrics_path)


def main():
    """Punto de entrada principal para entrenamiento."""
    setup_logging()
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

    _save_metrics_summary(history, Path(METRICS_DIR))


if __name__ == "__main__":
    main()
