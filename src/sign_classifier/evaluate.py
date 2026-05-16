import logging

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Model,
    val_path: str,
) -> dict:
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    logger.info("Generando predicciones...")
    predictions = model.predict(val_generator, verbose=1)
    y_true = val_generator.classes
    y_pred = np.argmax(predictions, axis=1)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES)

    logger.info("\n" + report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
