import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import Model

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODEL_PATH,
    VALIDATION_DATA_PATH,
    setup_logging,
)
from .data_loader import create_validation_dataset
from .exceptions import ModelError
from .predict import load_model_safe

logger = logging.getLogger(__name__)


def _collect_true_labels(val_ds) -> np.ndarray:
    labels = []
    for _, label_batch in val_ds:
        labels.append(np.argmax(label_batch.numpy(), axis=1))
    return np.concatenate(labels)


def evaluate_model(model: Model, val_path: str) -> dict:
    if model is None:
        raise ModelError("Se requiere un modelo cargado para evaluar")

    val_ds, class_names = create_validation_dataset(
        val_path=val_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=32,
    )

    class_names_sorted = sorted(class_names)

    logger.info("Generando predicciones...")
    predictions = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = _collect_true_labels(val_ds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names_sorted, zero_division=0
    )

    logger.info("\n" + report)
    logger.info("Matriz de confusión:\n%s", cm)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main():
    """Punto de entrada principal para evaluación."""
    setup_logging()
    logger.info("Iniciando evaluación del clasificador de señas...")
    logger.info("Modelo: %s", MODEL_PATH)
    logger.info("Datos de validación: %s", VALIDATION_DATA_PATH)

    model = load_model_safe(MODEL_PATH)
    metrics = evaluate_model(model, VALIDATION_DATA_PATH)

    logger.info("Accuracy:  %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall:    %.4f", metrics["recall"])
    logger.info("F1-Score:  %.4f", metrics["f1_score"])


if __name__ == "__main__":
    main()
