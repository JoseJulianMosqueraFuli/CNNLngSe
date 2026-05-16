import logging

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from .exceptions import PredictionError

logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise PredictionError(
            "target_size debe ser una tupla de 2 elementos (height, width)"
        )

    try:
        img = load_img(image_path, target_size=target_size)
    except FileNotFoundError as exc:
        raise PredictionError(f"Imagen no encontrada: {image_path}") from exc
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
