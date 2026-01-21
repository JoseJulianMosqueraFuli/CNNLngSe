"""
Tests de propiedades para el módulo del modelo CNN.

Feature: mejora-clasificador-senas
"""

import pytest
from hypothesis import given, strategies as st, settings

from sign_classifier.model import create_model


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
    @settings(max_examples=100)
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
            f"Se esperaban 3 capas Conv2D, se encontraron {len(conv_layers)}"
        )

        # Verificar filtros progresivos
        expected_filters = [32, 64, 128]
        actual_filters = [layer.filters for layer in conv_layers]
        assert actual_filters == expected_filters, (
            f"Filtros esperados {expected_filters}, encontrados {actual_filters}"
        )


    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100)
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
                f"Se esperaba BatchNormalization después de Conv2D en índice {idx}, "
                f"se encontró {next_layer}"
            )

    @given(
        height=st.integers(min_value=64, max_value=256),
        width=st.integers(min_value=64, max_value=256),
        num_classes=st.integers(min_value=2, max_value=100)
    )
    @settings(max_examples=100)
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
    @settings(max_examples=100)
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
