"""
Módulo de configuración para el clasificador de señas.

Centraliza todas las constantes y parámetros del proyecto.
Los valores pueden ser sobrescritos mediante variables de entorno con el
prefijo ``SIGN_CLASSIFIER_``. Por ejemplo:

    SIGN_CLASSIFIER_EPOCHS=50
    SIGN_CLASSIFIER_BATCH_SIZE=16
    SIGN_CLASSIFIER_LEARNING_RATE=0.001
"""

import logging
import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


class Settings(BaseSettings):
    """Configuración del proyecto cargada desde variables de entorno."""

    model_config = SettingsConfigDict(
        env_prefix="SIGN_CLASSIFIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Configuración de imágenes
    image_width: int = 150
    image_height: int = 150
    image_channels: int = 3

    # Configuración de entrenamiento
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0004
    use_transfer_learning: bool = False
    transfer_learning_backbone: str = "mobilenet_v3"

    # Rutas
    training_data_path: str = "./data/entrenamiento"
    validation_data_path: str = "./data/validacion"
    model_path: str = "./modelo/modelo.keras"
    log_dir: str = "./logs"
    metrics_dir: str = "./metrics"

    # Clases
    classes: list[str] = ["a", "b", "c"]

    def model_post_init(self, __context: object) -> None:
        # Normalizar rutas para que sean absolutas respecto al directorio de
        # trabajo actual, evitando inconsistencias cuando se ejecuta desde
        # distintos lugares.
        for attr in (
            "training_data_path",
            "validation_data_path",
            "model_path",
            "log_dir",
            "metrics_dir",
        ):
            value = getattr(self, attr)
            if not Path(value).is_absolute():
                setattr(self, attr, str(Path(value).resolve()))


try:
    _settings = Settings()
except ValidationError as exc:
    raise ConfigurationError(f"Configuración inválida: {exc}") from exc


# Configuración de imágenes
IMAGE_WIDTH = _settings.image_width
IMAGE_HEIGHT = _settings.image_height
IMAGE_CHANNELS = _settings.image_channels
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Configuración de entrenamiento
EPOCHS = _settings.epochs
BATCH_SIZE = _settings.batch_size
LEARNING_RATE = _settings.learning_rate
USE_TRANSFER_LEARNING = _settings.use_transfer_learning
TRANSFER_LEARNING_BACKBONE = _settings.transfer_learning_backbone

# Rutas
TRAINING_DATA_PATH = _settings.training_data_path
VALIDATION_DATA_PATH = _settings.validation_data_path
MODEL_PATH = _settings.model_path
LOG_DIR = _settings.log_dir
METRICS_DIR = _settings.metrics_dir

# Clases
CLASSES = _settings.classes
NUM_CLASSES = len(CLASSES)
