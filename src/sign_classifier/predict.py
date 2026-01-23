"""
Módulo de predicción para el clasificador de señas.

Expone funciones puras para cargar, preprocesar imágenes
y realizar predicciones.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_and_preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """
    Carga y preprocesa una imagen para predicción.

    Args:
        image_path: Ruta a la imagen a procesar
        target_size: Tupla (height, width) para redimensionar la imagen

    Returns:
        Array numpy con shape (1, height, width, 3) y valores
        normalizados [0, 1]

    Raises:
        FileNotFoundError: Si la imagen no existe
        ValueError: Si el formato de imagen es inválido o
            target_size es incorrecto
    """
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError(
            "target_size debe ser una tupla de 2 elementos (height, width)"
        )

    try:
        # Cargar imagen con el tamaño objetivo
        img = load_img(image_path, target_size=target_size)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}") from exc
    except Exception as e:
        raise ValueError(
            f"Formato de imagen inválido. "
            f"Formatos soportados: JPEG, PNG, BMP, GIF. Error: {str(e)}"
        ) from e

    # Convertir a array numpy
    img_array = img_to_array(img)

    # Normalizar valores a [0, 1]
    img_array = img_array / 255.0

    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_class(model: Model, image: np.ndarray, classes: list) -> str:
    """
    Predice la clase de una imagen preprocesada.

    Args:
        model: Modelo Keras cargado
        image: Array numpy preprocesado con shape (1, height, width, 3)
        classes: Lista de nombres de clases

    Returns:
        Nombre de la clase predicha

    Raises:
        ValueError: Si la imagen o clases son inválidas
    """
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError(
            "classes debe ser una lista no vacía de nombres de clases"
        )

    if not isinstance(image, np.ndarray):
        raise ValueError("image debe ser un array numpy")

    if len(image.shape) != 4 or image.shape[0] != 1:
        raise ValueError(
            f"image debe tener shape (1, height, width, channels), "
            f"recibido {image.shape}"
        )

    # Realizar predicción
    predictions = model.predict(image, verbose=0)

    # Obtener índice de la clase con mayor probabilidad
    predicted_index = np.argmax(predictions[0])

    # Validar que el índice está dentro del rango de clases
    if predicted_index >= len(classes):
        raise ValueError(
            f"Índice predicho {predicted_index} fuera del rango de clases "
            f"(0-{len(classes)-1})"
        )

    return classes[predicted_index]
