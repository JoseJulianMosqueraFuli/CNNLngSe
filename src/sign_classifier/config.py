"""
Módulo de configuración para el clasificador de señas.

Centraliza todas las constantes y parámetros del proyecto.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# Configuración de imágenes
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Configuración de entrenamiento
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0004

# Rutas
TRAINING_DATA_PATH = "./data/entrenamiento"
VALIDATION_DATA_PATH = "./data/validacion"
MODEL_PATH = "./modelo/modelo.keras"
LOG_DIR = "./logs"

# Clases
CLASSES = ["a", "b", "c"]
NUM_CLASSES = len(CLASSES)
