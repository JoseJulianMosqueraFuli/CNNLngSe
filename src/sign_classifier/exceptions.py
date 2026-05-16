class SignClassifierError(Exception):
    """Error base del clasificador de señas."""


class DataLoadingError(SignClassifierError):
    """Error al cargar datos de entrenamiento o validación."""


class ModelError(SignClassifierError):
    """Error relacionado con el modelo (creación, carga, predicción)."""


class PredictionError(SignClassifierError):
    """Error durante la predicción."""


class ConfigurationError(SignClassifierError):
    """Error en la configuración del proyecto."""
