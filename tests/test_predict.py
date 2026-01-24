"""
Tests para el módulo de predicción.

Incluye tests unitarios básicos y tests de propiedades.

Feature: mejora-clasificador-senas
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from hypothesis import given, strategies as st, settings, assume

from sign_classifier.predict import load_and_preprocess_image, predict_class
from sign_classifier.model import create_model
from sign_classifier.config import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES


# =============================================================================
# Tests Unitarios Básicos
# =============================================================================

class TestPreprocessingUnitTests:
    """Tests unitarios básicos para el preprocesamiento de imágenes."""

    def test_load_and_preprocess_image_with_valid_image(self):
        """Verifica que una imagen válida se preprocesa correctamente."""
        # Crear imagen temporal
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            temp_path = f.name
        
        try:
            result = load_and_preprocess_image(temp_path, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
            assert result.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            assert result.dtype == np.float32 or result.dtype == np.float64
            assert np.all(result >= 0.0) and np.all(result <= 1.0)
        finally:
            os.unlink(temp_path)

    def test_load_and_preprocess_image_file_not_found(self):
        """Verifica que se lanza error cuando la imagen no existe."""
        with pytest.raises(FileNotFoundError, match="Imagen no encontrada"):
            load_and_preprocess_image("/ruta/inexistente/imagen.png", (150, 150))

    def test_load_and_preprocess_image_invalid_target_size_not_tuple(self):
        """Verifica que se lanza error con target_size no tupla."""
        with pytest.raises(ValueError, match="target_size debe ser una tupla"):
            load_and_preprocess_image("test.png", [150, 150])

    def test_load_and_preprocess_image_invalid_target_size_wrong_length(self):
        """Verifica que se lanza error con target_size de longitud incorrecta."""
        with pytest.raises(ValueError, match="target_size debe ser una tupla de 2 elementos"):
            load_and_preprocess_image("test.png", (150,))


class TestPredictionUnitTests:
    """Tests unitarios básicos para la predicción."""

    def test_predict_class_returns_valid_class(self):
        """Verifica que predict_class retorna una clase válida."""
        model = create_model((64, 64, 3), 3)
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)
        classes = ["a", "b", "c"]
        
        result = predict_class(model, image, classes)
        
        assert result in classes

    def test_predict_class_invalid_classes_empty(self):
        """Verifica que se lanza error con lista de clases vacía."""
        model = create_model((64, 64, 3), 3)
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="classes debe ser una lista no vacía"):
            predict_class(model, image, [])

    def test_predict_class_invalid_classes_not_list(self):
        """Verifica que se lanza error con clases no lista."""
        model = create_model((64, 64, 3), 3)
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="classes debe ser una lista no vacía"):
            predict_class(model, image, "abc")

    def test_predict_class_invalid_image_not_array(self):
        """Verifica que se lanza error con imagen no array."""
        model = create_model((64, 64, 3), 3)
        classes = ["a", "b", "c"]
        
        with pytest.raises(ValueError, match="image debe ser un array numpy"):
            predict_class(model, "not_an_array", classes)

    def test_predict_class_invalid_image_wrong_shape(self):
        """Verifica que se lanza error con imagen de shape incorrecto."""
        model = create_model((64, 64, 3), 3)
        image = np.random.rand(64, 64, 3).astype(np.float32)  # Sin dimensión batch
        classes = ["a", "b", "c"]
        
        with pytest.raises(ValueError, match="image debe tener shape"):
            predict_class(model, image, classes)


# =============================================================================
# Tests de Propiedades
# =============================================================================


class TestImagePreprocessingProperty:
    """
    Property 3: Preprocesamiento de Imagen Consistente

    Para cualquier imagen válida, la función load_and_preprocess_image()
    debe retornar un array numpy con shape (1, height, width, 3) y valores
    normalizados entre 0 y 1.

    **Validates: Requirements 6.3**
    """

    @given(
        height=st.integers(min_value=32, max_value=300),
        width=st.integers(min_value=32, max_value=300),
        target_height=st.integers(min_value=32, max_value=256),
        target_width=st.integers(min_value=32, max_value=256)
    )
    @settings(max_examples=100, deadline=None)
    def test_preprocessed_image_has_correct_shape(
        self, height, width, target_height, target_width
    ):
        """
        Verifica que la imagen preprocesada tiene shape
        (1, target_height, target_width, 3).

        **Validates: Requirements 6.3**
        """
        # Crear imagen temporal de prueba
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            temp_path = f.name

        try:
            target_size = (target_height, target_width)
            result = load_and_preprocess_image(temp_path, target_size)

            # Verificar shape
            expected_shape = (1, target_height, target_width, 3)
            assert result.shape == expected_shape, (
                f"Shape esperado {expected_shape}, obtenido {result.shape}"
            )
        finally:
            os.unlink(temp_path)

    @given(
        height=st.integers(min_value=32, max_value=200),
        width=st.integers(min_value=32, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_preprocessed_image_values_normalized(self, height, width):
        """
        Verifica que los valores de la imagen están normalizados entre 0 y 1.

        **Validates: Requirements 6.3**
        """
        # Crear imagen temporal con valores variados
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            temp_path = f.name

        try:
            target_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
            result = load_and_preprocess_image(temp_path, target_size)

            # Verificar que todos los valores están en [0, 1]
            assert np.all(result >= 0.0), (
                f"Valores mínimos deben ser >= 0, encontrado {result.min()}"
            )
            assert np.all(result <= 1.0), (
                f"Valores máximos deben ser <= 1, encontrado {result.max()}"
            )
        finally:
            os.unlink(temp_path)

    @given(
        height=st.integers(min_value=32, max_value=200),
        width=st.integers(min_value=32, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_preprocessed_image_is_numpy_array(self, height, width):
        """
        Verifica que el resultado es un array numpy.

        **Validates: Requirements 6.3**
        """
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            temp_path = f.name

        try:
            target_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
            result = load_and_preprocess_image(temp_path, target_size)

            assert isinstance(result, np.ndarray), (
                f"Resultado debe ser np.ndarray, obtenido {type(result)}"
            )
        finally:
            os.unlink(temp_path)


class TestDeterministicPredictionProperty:
    """
    Property 4: Predicción Determinista

    Para cualquier imagen preprocesada y modelo cargado, la función
    predict_class() debe retornar una clase válida del conjunto de clases
    definido, y llamadas repetidas con la misma entrada deben producir
    el mismo resultado.

    **Validates: Requirements 4.5, 6.3**
    """

    @given(
        num_classes=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    def test_prediction_returns_valid_class(self, num_classes):
        """
        Verifica que la predicción retorna una clase válida del conjunto.

        **Validates: Requirements 4.5, 6.3**
        """
        # Crear modelo y clases
        input_shape = (64, 64, 3)
        model = create_model(input_shape, num_classes)
        classes = [f"class_{i}" for i in range(num_classes)]

        # Crear imagen de prueba normalizada
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)

        result = predict_class(model, image, classes)

        # Verificar que el resultado es una clase válida
        assert result in classes, (
            f"Clase predicha '{result}' no está en {classes}"
        )

    @given(
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=20, deadline=None)
    def test_prediction_is_deterministic(self, seed):
        """
        Verifica que llamadas repetidas producen el mismo resultado.

        **Validates: Requirements 4.5, 6.3**
        """
        # Usar seed para reproducibilidad
        np.random.seed(seed)

        # Crear modelo y clases
        input_shape = (64, 64, 3)
        num_classes = 3
        model = create_model(input_shape, num_classes)
        classes = ["a", "b", "c"]

        # Crear imagen de prueba fija
        np.random.seed(42)  # Seed fijo para la imagen
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)

        # Realizar múltiples predicciones
        result1 = predict_class(model, image, classes)
        result2 = predict_class(model, image, classes)
        result3 = predict_class(model, image, classes)

        # Verificar que todas las predicciones son iguales
        assert result1 == result2 == result3, (
            f"Predicciones no deterministas: {result1}, {result2}, {result3}"
        )

    @given(
        height=st.integers(min_value=32, max_value=128),
        width=st.integers(min_value=32, max_value=128)
    )
    @settings(max_examples=20, deadline=None)
    def test_prediction_with_different_image_sizes(self, height, width):
        """
        Verifica que la predicción funciona con diferentes tamaños de imagen.

        **Validates: Requirements 4.5, 6.3**
        """
        # Crear modelo
        input_shape = (height, width, 3)
        num_classes = len(CLASSES)
        model = create_model(input_shape, num_classes)

        # Crear imagen de prueba con el tamaño correcto
        image = np.random.rand(1, height, width, 3).astype(np.float32)

        result = predict_class(model, image, CLASSES)

        # Verificar que el resultado es una clase válida
        assert result in CLASSES, (
            f"Clase predicha '{result}' no está en {CLASSES}"
        )
