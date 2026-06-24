"""
Descarga el dataset ASL Alphabet de Kaggle y filtra las clases A, B, C.

Requisitos:
    1. Tener una cuenta en Kaggle.
    2. Generar un API token en https://www.kaggle.com/settings/account
    3. Guardar el archivo kaggle.json en ~/.kaggle/kaggle.json

Uso:
    poetry run python -m sign_classifier.download_asl_alphabet \
        --output-dir ./data_kaggle \
        --classes a b c \
        --split-ratio 0.8
"""

import argparse
import logging
import shutil
from pathlib import Path

from .config import setup_logging

logger = logging.getLogger(__name__)

DATASET_HANDLE = "grassknoted/asl-alphabet"


def download_asl_alphabet(output_dir: str) -> str:
    """
    Descarga el dataset ASL Alphabet de Kaggle.

    Args:
        output_dir: Directorio donde se descargará el dataset.

    Returns:
        Ruta al directorio descomprimido del dataset.
    """
    import kagglehub

    logger.info("Descargando dataset de Kaggle: %s", DATASET_HANDLE)
    path = kagglehub.dataset_download(
        DATASET_HANDLE,
        path=output_dir,
    )
    logger.info("Dataset descargado en: %s", path)
    return path


def filter_classes(
    source_dir: str,
    output_dir: str,
    classes: list[str],
    split_ratio: float,
) -> None:
    """
    Copia solo las clases seleccionadas del dataset descargado.

    Args:
        source_dir: Directorio con el dataset descargado.
        output_dir: Directorio de salida organizado por train/val.
        classes: Lista de clases a conservar.
        split_ratio: Proporción de imágenes para entrenamiento.
    """
    source_path = Path(source_dir)

    # El dataset de Kaggle suele tener estructura asl_alphabet_train/asl_alphabet_test
    # Buscamos la carpeta que contenga subdirectorios por clase.
    data_dirs = [
        source_path / "asl_alphabet_train",
        source_path / "asl_alphabet_test",
        source_path,
    ]

    train_dir = None
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        class_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        if all(cls in class_dirs for cls in classes):
            train_dir = data_dir
            break

    if train_dir is None:
        raise FileNotFoundError(
            f"No se encontró el directorio con las clases {classes} en {source_dir}"
        )

    output_path = Path(output_dir)
    train_output = output_path / "entrenamiento"
    val_output = output_path / "validacion"

    for cls in classes:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            logger.warning("Clase %s no encontrada en el dataset", cls)
            continue

        images = sorted(cls_dir.iterdir())
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            if img.is_file():
                dest = train_output / cls / img.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dest)

        for img in val_images:
            if img.is_file():
                dest = val_output / cls / img.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dest)

        logger.info(
            "Clase %s: %d train, %d val copiadas",
            cls,
            len(train_images),
            len(val_images),
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Descargar dataset ASL Alphabet de Kaggle y filtrar clases"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_kaggle",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["a", "b", "c"],
        help="Clases a descargar",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Proporción de imágenes para entrenamiento",
    )
    args = parser.parse_args()

    downloaded_path = download_asl_alphabet(args.output_dir)
    filter_classes(
        source_dir=downloaded_path,
        output_dir=args.output_dir,
        classes=args.classes,
        split_ratio=args.split_ratio,
    )

    logger.info("Dataset listo en: %s", args.output_dir)


if __name__ == "__main__":
    main()
