"""
Módulo para exportar modelos entrenados a formatos de despliegue.

Soporta:
- SavedModel (para TensorFlow Serving)
- TensorFlow Lite (para móvil/edge)
"""

import argparse
import logging
from pathlib import Path

from .config import MODEL_PATH, setup_logging
from .predict import load_model_safe

logger = logging.getLogger(__name__)


def export_to_savedmodel(model_path: str, output_dir: str) -> None:
    """
    Exporta un modelo entrenado al formato SavedModel.

    Args:
        model_path: Ruta al modelo .keras.
        output_dir: Directorio de salida para el SavedModel.
    """
    model = load_model_safe(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir, save_format="tf")
    logger.info("Modelo exportado a SavedModel en: %s", output_dir)


def export_to_tflite(model_path: str, output_path: str) -> None:
    """
    Exporta un modelo entrenado al formato TensorFlow Lite.

    Args:
        model_path: Ruta al modelo .keras.
        output_path: Ruta del archivo .tflite de salida.
    """
    import tensorflow as tf

    model = load_model_safe(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(tflite_model)

    logger.info("Modelo exportado a TensorFlow Lite en: %s", output_path)


def main() -> None:
    """Punto de entrada principal para exportación de modelos."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Exportar modelo entrenado a SavedModel o TensorFlow Lite"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Ruta al modelo .keras",
    )
    parser.add_argument(
        "--savedmodel-dir",
        type=str,
        default="./modelo_savedmodel",
        help="Directorio de salida para SavedModel",
    )
    parser.add_argument(
        "--tflite-path",
        type=str,
        default="./modelo.tflite",
        help="Ruta de salida para TensorFlow Lite",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["savedmodel", "tflite", "both"],
        default="both",
        help="Formato de exportación",
    )
    args = parser.parse_args()

    if args.format in ("savedmodel", "both"):
        export_to_savedmodel(args.model_path, args.savedmodel_dir)
    if args.format in ("tflite", "both"):
        export_to_tflite(args.model_path, args.tflite_path)


if __name__ == "__main__":
    main()
