import argparse
import logging
import os

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from .config import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH, setup_logging
from .exceptions import PredictionError

logger = logging.getLogger(__name__)

MAX_IMAGE_FILE_SIZE_MB = 10


def load_model_safe(model_path: str):
    """
    Carga un modelo Keras en modo seguro.

    El modo seguro deshabilita la ejecución de código arbitrario que podría
    estar embebido en archivos de modelo manipulados.
    """
    try:
        return load_model(model_path, safe_mode=True)
    except Exception as exc:
        raise PredictionError(f"No se pudo cargar el modelo: {model_path}") from exc


def load_and_preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise PredictionError(
            "target_size debe ser una tupla de 2 elementos (height, width)"
        )

    try:
        file_size = os.path.getsize(image_path)
    except FileNotFoundError as exc:
        raise PredictionError(f"Imagen no encontrada: {image_path}") from exc
    except OSError as exc:
        raise PredictionError(f"No se pudo leer la imagen: {image_path}") from exc

    max_size_bytes = MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise PredictionError(
            f"Imagen demasiado grande: {file_size / (1024 * 1024):.2f} MB. "
            f"Máximo permitido: {MAX_IMAGE_FILE_SIZE_MB} MB."
        )

    try:
        img = load_img(image_path, target_size=target_size)
    except Exception as e:
        raise PredictionError(
            f"Formato de imagen inválido: {image_path}. Error: {e}"
        ) from e

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_class(model: Model, image: np.ndarray, classes: list) -> str:
    if not isinstance(classes, list) or len(classes) == 0:
        raise PredictionError(
            "classes debe ser una lista no vacía de nombres de clases"
        )

    if not isinstance(image, np.ndarray):
        raise PredictionError("image debe ser un array numpy")

    if len(image.shape) != 4 or image.shape[0] != 1:
        raise PredictionError(
            f"image debe tener shape (1, height, width, channels), "
            f"recibido {image.shape}"
        )

    predictions = model.predict(image, verbose=0)
    predicted_index = np.argmax(predictions[0])

    if predicted_index >= len(classes):
        raise PredictionError(
            f"Índice predicho {predicted_index} fuera del rango de clases "
            f"(0-{len(classes) - 1})"
        )

    return classes[predicted_index]


def main():
    """Punto de entrada principal para predicción de una imagen."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Predecir clase de una imagen")
    parser.add_argument("image", type=str, help="Ruta a la imagen a clasificar")
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Ruta al modelo entrenado",
    )
    args = parser.parse_args()

    logger.info("Cargando modelo desde %s", args.model_path)
    model = load_model_safe(args.model_path)

    logger.info("Preprocesando imagen %s", args.image)
    image = load_and_preprocess_image(args.image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    predicted_class = predict_class(model, image, CLASSES)
    logger.info("Clase predicha: %s", predicted_class)
    print(predicted_class)


if __name__ == "__main__":
    main()
