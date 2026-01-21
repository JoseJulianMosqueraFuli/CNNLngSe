"""
Módulo de configuración para el clasificador de señas.

Centraliza todas las constantes y parámetros del proyecto.
"""

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

# Clases
CLASSES = ["a", "b", "c"]
NUM_CLASSES = len(CLASSES)
