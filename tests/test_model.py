"""
Tests para el módulo del modelo CNN.

Incluye tests unitarios básicos y tests de propiedades.

Feature: mejora-clasificador-senas
"""

import pytest
from hypothesis import given, strategies as st, settings

from sign_classifier.model import create_model
from sign_classifier.config import IMAGE_SHAPE, NUM_CLASSES


# =============================================================================
# Tests Unitarios Básicos
# =============================================================================

class TestModelUnitTests:
    """Tests unitarios básicos para el modelo CNN."""

    def test_create_model_with_default_config(self):
        """Verifica que el modelo se crea correctamente con la configuración por defecto."""
        model = create_model(IMAGE_SHAPE, NUM_CLASSES)
        
        assert model is not None
        assert model.input_shape == (None, *IMAGE_SHAPE)
        assert model.output_shape == (None, NUM_CLASSES)

    def test_create_model_layer_count(self):
        """Verifica que el modelo tiene el número correcto de capas."""
        model = create_model(IMAGE_SHAPE, NUM_CLASSES)
        
        # 3 Conv2D + 3 BatchNorm + 3 MaxPool + Flatten + 2 Dense + 2 Dropout + 1 Dense output = 15
        assert len(model.layers) == 15

    def test_create_model_is_compiled(self):
        """Verifica que el modelo está compilado."""
        model = create_model(IMAGE_SHAPE, NUM_CLASSES)
        
        assert model.optimizer is not None
        assert model.loss is not None

    def test_create_model_invalid_input_shape_not_tuple(self):
        """Verifica que se lanza error con input_shape inválido (no tupla)."""
        with pytest.raises(ValueError, match="input_shape debe ser una tupla"):
            create_model([150, 150, 3], NUM_CLASSES)

    def test_create_model_invalid_input_shape_wrong_length(self):
        """Verifica que se lanza error con input_shape de longitud incorrecta."""
        with pytest.raises(ValueError, match="input_shape debe ser una tupla de 3 elementos"):
            create_model((150, 150), NUM_CLASSES)

    def test_create_model_invalid_num_classes_not_int(self):
        """Verifica que se lanza error con num_classes no entero."""
        with pytest.raises(ValueError, match="num_classes debe ser un entero"):
            create_model(IMAGE_SHAPE, "3")

    def test_create_model_invalid_num_classes_less_than_two(self):
        """Verifica que se lanza error con num_classes < 2."""
        with pytest.raises(ValueError, match="num_classes debe ser un entero >= 2"):
            create_model(IMAGE_SHAPE, 1)


# =============================================================================
# Tests de Propiedades
# =============================================================================


class TestModelStructureProperty:
    """
    Property 1: Estructura del Modelo CNN

    Para cualquier modelo creado con create_model(), el modelo debe contener
    exactamente 3 bloques convolucionales con filtros progresivos (32, 64, 128),
    cada bloque convolucional debe ser seguido por una capa BatchNormalization,
    debe incluir capas Dropout en la sección densa, y la capa final debe usar
    activación Softmax.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_model_has_three_conv_blocks_with_progressive_filters(
        self, height, width, num_classes
    ):
        """
        Verifica que el modelo tiene 3 bloques convolucionales con filtros
        progresivos (32, 64, 128).

        **Validates: Requirements 3.1**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Extraer capas Conv2D
        conv_layers = [
            layer for layer in model.layers
            if layer.__class__.__name__ == 'Conv2D'
        ]

        # Debe haber exactamente 3 capas Conv2D
        assert len(conv_layers) == 3, (
            f"Se esperaban 3 capas Conv2D, encontradas {len(conv_layers)}"
        )

        # Verificar filtros progresivos
        expected_filters = [32, 64, 128]
        actual_filters = [layer.filters for layer in conv_layers]
        assert actual_filters == expected_filters, (
            f"Filtros esperados {expected_filters}, "
            f"encontrados {actual_filters}"
        )


    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_batch_normalization_after_each_conv(self, height, width, num_classes):
        """
        Verifica que cada capa Conv2D es seguida por BatchNormalization.

        **Validates: Requirements 3.2**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        layer_names = [layer.__class__.__name__ for layer in model.layers]

        # Encontrar índices de Conv2D
        conv_indices = [
            i for i, name in enumerate(layer_names) if name == 'Conv2D'
        ]

        # Verificar que después de cada Conv2D hay BatchNormalization
        for idx in conv_indices:
            next_layer = layer_names[idx + 1]
            assert next_layer == 'BatchNormalization', (
                f"Se esperaba BatchNormalization después de Conv2D "
                f"en índice {idx}, se encontró {next_layer}"
            )

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_dropout_in_dense_section(self, height, width, num_classes):
        """
        Verifica que hay capas Dropout en la sección densa.

        **Validates: Requirements 3.3**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Contar capas Dropout
        dropout_layers = [
            layer for layer in model.layers
            if layer.__class__.__name__ == 'Dropout'
        ]

        # Debe haber al menos una capa Dropout
        assert len(dropout_layers) >= 1, (
            "El modelo debe incluir al menos una capa Dropout"
        )

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_relu_and_softmax_activations(self, height, width, num_classes):
        """
        Verifica que las capas ocultas usan ReLU y la capa final usa Softmax.

        **Validates: Requirements 3.4**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Verificar capas Conv2D usan ReLU
        conv_layers = [
            layer for layer in model.layers
            if layer.__class__.__name__ == 'Conv2D'
        ]
        for layer in conv_layers:
            activation = layer.activation.__name__
            assert activation == 'relu', (
                f"Conv2D debe usar ReLU, encontrado {activation}"
            )

        # Verificar capas Dense (excepto la última) usan ReLU
        dense_layers = [
            layer for layer in model.layers
            if layer.__class__.__name__ == 'Dense'
        ]

        for layer in dense_layers[:-1]:
            activation = layer.activation.__name__
            assert activation == 'relu', (
                f"Dense oculta debe usar ReLU, encontrado {activation}"
            )

        # Verificar última capa Dense usa Softmax
        last_dense = dense_layers[-1]
        activation = last_dense.activation.__name__
        assert activation == 'softmax', (
            f"Última capa Dense debe usar Softmax, encontrado {activation}"
        )


class TestCompiledModelProperty:
    """
    Property 2: Modelo Compilado Válido

    Para cualquier llamada a create_model() con input_shape y num_classes válidos,
    la función debe retornar un modelo Keras compilado con optimizer, loss function
    y metrics configurados correctamente.

    **Validates: Requirements 4.4**
    """

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_model_is_compiled_with_optimizer(self, height, width, num_classes):
        """
        Verifica que el modelo tiene un optimizer configurado.

        **Validates: Requirements 4.4**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Verificar que el modelo tiene optimizer
        assert model.optimizer is not None, (
            "El modelo debe tener un optimizer configurado"
        )

        # Verificar que es Adam
        optimizer_name = model.optimizer.__class__.__name__
        assert optimizer_name == 'Adam', (
            f"Se esperaba optimizer Adam, encontrado {optimizer_name}"
        )

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_model_is_compiled_with_loss_function(self, height, width, num_classes):
        """
        Verifica que el modelo tiene una loss function configurada.

        **Validates: Requirements 4.4**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Verificar que el modelo tiene loss configurado
        assert model.loss is not None, (
            "El modelo debe tener una loss function configurada"
        )

        # Verificar que es categorical_crossentropy
        loss_name = model.loss
        assert loss_name == 'categorical_crossentropy', (
            f"Se esperaba loss categorical_crossentropy, encontrado {loss_name}"
        )

    @given(
        height=st.integers(min_value=64, max_value=128),
        width=st.integers(min_value=64, max_value=128),
        num_classes=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=10, deadline=None)
    def test_model_is_compiled_with_metrics(self, height, width, num_classes):
        """
        Verifica que el modelo tiene metrics configurados.

        **Validates: Requirements 4.4**
        """
        input_shape = (height, width, 3)
        model = create_model(input_shape, num_classes)

        # Verificar que el modelo está compilado
        assert model.compiled, (
            "El modelo debe estar compilado"
        )

        # Obtener configuración de compilación
        compile_config = model.get_compile_config()

        # Verificar que metrics está configurado
        assert 'metrics' in compile_config, (
            "El modelo debe tener metrics en su configuración"
        )

        # Verificar que accuracy está en las métricas
        metrics = compile_config['metrics']
        assert 'accuracy' in metrics, (
            f"Se esperaba metric 'accuracy', encontradas {metrics}"
        )
