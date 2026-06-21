"""
Módulo de predicción por lotes para el clasificador de señas.

Permite procesar múltiples imágenes de una carpeta y exportar los resultados
a un archivo CSV.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model

from .config import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH, setup_logging
from .predict import load_and_preprocess_image, load_model_safe, predict_class

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def discover_images(
    directory: str, extensions: tuple = DEFAULT_EXTENSIONS
) -> list[str]:
    """
    Descubre todas las imágenes soportadas en un directorio.

    Args:
        directory: Ruta al directorio a explorar.
        extensions: Extensiones de archivo a considerar.

    Returns:
        Lista de rutas de imágenes encontradas.
    """
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"La ruta no es un directorio: {directory}")

    images = sorted(
        str(path)
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    logger.info("Descubiertas %d imágenes en %s", len(images), directory)
    return images


def predict_batch(
    model: Model,
    image_paths: list[str],
    classes: list[str],
    target_size: tuple = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> list[dict]:
    """
    Predice la clase de un lote de imágenes.

    Args:
        model: Modelo Keras cargado.
        image_paths: Lista de rutas de imágenes.
        classes: Lista de nombres de clases.
        target_size: Tamaño al que redimensionar las imágenes.

    Returns:
        Lista de diccionarios con los resultados de cada imagen.
    """
    results = []
    for image_path in image_paths:
        try:
            image = load_and_preprocess_image(image_path, target_size)
            predicted_class = predict_class(model, image, classes)
            probabilities = model.predict(image, verbose=0)[0]
            confidence = float(np.max(probabilities))
        except Exception as exc:
            logger.error("Error procesando %s: %s", image_path, exc)
            predicted_class = "ERROR"
            confidence = 0.0
            probabilities = [0.0] * len(classes)

        row = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }
        for idx, class_name in enumerate(classes):
            row[f"prob_{class_name}"] = float(probabilities[idx])
        results.append(row)

    return results


def save_predictions_csv(results: list[dict], output_path: str) -> None:
    """
    Guarda los resultados de predicción en un archivo CSV.

    Args:
        results: Lista de diccionarios con los resultados.
        output_path: Ruta del archivo CSV de salida.
    """
    if not results:
        logger.warning("No hay resultados para guardar")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Predicciones guardadas en: %s", output_path)


def main() -> None:
    """Punto de entrada principal para predicción por lotes."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Predecir clases para un lote de imágenes"
    )
    parser.add_argument(
        "input_dir", type=str, help="Directorio con las imágenes a clasificar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./predictions.csv",
        help="Ruta del archivo CSV de salida",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Ruta al modelo entrenado",
    )
    args = parser.parse_args()

    logger.info("Cargando modelo desde %s", args.model_path)
    model = load_model_safe(args.model_path)

    image_paths = discover_images(args.input_dir)
    if not image_paths:
        logger.warning("No se encontraron imágenes en %s", args.input_dir)
        return

    logger.info("Generando predicciones para %d imágenes...", len(image_paths))
    results = predict_batch(model, image_paths, CLASSES)
    save_predictions_csv(results, args.output)


if __name__ == "__main__":
    main()
