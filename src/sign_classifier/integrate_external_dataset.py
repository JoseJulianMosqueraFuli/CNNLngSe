"""
Integración de datasets externos de ASL.

Este script toma un directorio con imágenes organizadas por clase (por ejemplo,
descargado manualmente de Kaggle, Roboflow o cualquier fuente pública) y las
integra en la estructura de datos del proyecto.

La estructura esperada del directorio externo es:

    external_dataset/
    ├── a/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── b/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── c/
        └── ...

Uso:
    poetry run python -m sign_classifier.integrate_external_dataset \
        --source-dir ./external_dataset \
        --target-dir ./data/entrenamiento \
        --split-ratio 0.8
"""

import argparse
import logging
import shutil
from pathlib import Path

from .config import setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def integrate_dataset(
    source_dir: str,
    target_dir: str,
    split_ratio: float = 1.0,
    val_target_dir: str | None = None,
) -> None:
    """
    Copia imágenes de un dataset externo al directorio destino.

    Args:
        source_dir: Directorio externo organizado por clases.
        target_dir: Directorio destino (train).
        split_ratio: Proporción de imágenes que van a train (0.0 - 1.0).
            El resto va a val_target_dir si se especifica.
        val_target_dir: Directorio destino para validación. Si es None, todo
            va a target_dir.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Directorio fuente no encontrado: {source_dir}")

    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No se encontraron subdirectorios de clases en {source_dir}")

    total_copied = 0
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            logger.warning("No se encontraron imágenes en %s", class_dir)
            continue

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img_path in train_images:
            dest = target_path / class_dir.name / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)
            total_copied += 1

        if val_target_dir and val_images:
            val_path = Path(val_target_dir)
            for img_path in val_images:
                dest = val_path / class_dir.name / img_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest)
                total_copied += 1

        logger.info(
            "Clase %s: %d train, %d val copiadas",
            class_dir.name,
            len(train_images),
            len(val_images),
        )

    logger.info("Total de imágenes integradas: %d", total_copied)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Integrar dataset externo de ASL al proyecto"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directorio externo organizado por clases",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="./data/entrenamiento",
        help="Directorio destino para entrenamiento",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=1.0,
        help="Proporción de imágenes para entrenamiento (0.0 - 1.0)",
    )
    parser.add_argument(
        "--val-target-dir",
        type=str,
        default=None,
        help="Directorio destino para validación (opcional)",
    )
    args = parser.parse_args()

    integrate_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        split_ratio=args.split_ratio,
        val_target_dir=args.val_target_dir,
    )


if __name__ == "__main__":
    main()
