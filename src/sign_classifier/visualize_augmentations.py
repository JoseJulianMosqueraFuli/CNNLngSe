"""
Visualización de imágenes aumentadas.

Este script genera una cuadrícula comparativa por clase: imagen original junto
a varias variantes generadas por el módulo de augmentación. Es útil para
verificar que las transformaciones mantienen el sentido de la seña.

Uso:
    poetry run python -m sign_classifier.visualize_augmentations \
        --input-dir ./data/entrenamiento \
        --output-path ./augmentation_preview.png \
        --variants-per-class 6
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .generate_augmented_data import augment_image, discover_images
from .config import setup_logging

logger = logging.getLogger(__name__)


def visualize_augmentations(
    input_dir: str,
    output_path: str,
    variants_per_class: int,
) -> None:
    """
    Genera una cuadrícula con la imagen original y variantes aumentadas por clase.

    Args:
        input_dir: Directorio con subcarpetas por clase.
        output_path: Ruta donde guardar la imagen de visualización.
        variants_per_class: Número de variantes a mostrar por clase.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {input_dir}")

    class_dirs = sorted(d for d in input_path.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"No se encontraron subdirectorios de clases en {input_dir}")

    num_classes = len(class_dirs)
    cols = variants_per_class + 1  # original + variantes
    fig, axes = plt.subplots(
        num_classes,
        cols,
        figsize=(cols * 2.5, num_classes * 2.5),
        squeeze=False,
    )

    for row_idx, class_dir in enumerate(class_dirs):
        images = discover_images(class_dir)
        if not images:
            logger.warning("No se encontraron imágenes en %s", class_dir)
            continue

        # Usar la primera imagen como original
        original_path = images[0]
        original = Image.open(original_path).convert("RGB")

        axes[row_idx][0].imshow(original)
        axes[row_idx][0].set_title(f"{class_dir.name} - original")
        axes[row_idx][0].axis("off")

        for col_idx in range(1, cols):
            augmented = augment_image(original.copy())
            axes[row_idx][col_idx].imshow(augmented)
            axes[row_idx][col_idx].set_title(f"{class_dir.name} - aug {col_idx}")
            axes[row_idx][col_idx].axis("off")

    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info("Visualización guardada en: %s", output_file)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Visualizar imágenes aumentadas por clase"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/entrenamiento",
        help="Directorio con imágenes organizadas por clase",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./augmentation_preview.png",
        help="Ruta de salida de la imagen de visualización",
    )
    parser.add_argument(
        "--variants-per-class",
        type=int,
        default=6,
        help="Número de variantes a mostrar por clase",
    )
    args = parser.parse_args()

    visualize_augmentations(
        input_dir=args.input_dir,
        output_path=args.output_path,
        variants_per_class=args.variants_per_class,
    )


if __name__ == "__main__":
    main()
