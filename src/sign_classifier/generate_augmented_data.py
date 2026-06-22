"""
Generación de datos aumentados a partir de imágenes reales de señas.

Este script toma las imágenes existentes y genera variantes realistas usando
transformaciones geométricas y de color. Mantiene la coherencia del dataset
porque parte de imágenes reales de manos haciendo señas.

Uso:
    poetry run python -m sign_classifier.generate_augmented_data \
        --input-dir ./data/entrenamiento \
        --output-dir ./data/entrenamiento \
        --variants-per-image 20
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _apply_rotation(img: Image.Image, max_degrees: float = 15.0) -> Image.Image:
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _apply_zoom(img: Image.Image, scale_range: tuple = (0.9, 1.1)) -> Image.Image:
    scale = random.uniform(*scale_range)
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = img.resize((new_width, new_height), Image.BILINEAR)

    if scale >= 1.0:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    # Si se hizo zoom out, centramos la imagen sobre un fondo blanco.
    result = Image.new("RGB", (width, height), (255, 255, 255))
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    result.paste(resized, (left, top))
    return result


def _apply_translation(
    img: Image.Image, max_shift_ratio: float = 0.1
) -> Image.Image:
    width, height = img.size
    max_dx = int(width * max_shift_ratio)
    max_dy = int(height * max_shift_ratio)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    result = Image.new("RGB", (width, height), (255, 255, 255))
    result.paste(img, (dx, dy))
    return result


def _apply_brightness(
    img: Image.Image, factor_range: tuple = (0.7, 1.3)
) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def _apply_contrast(img: Image.Image, factor_range: tuple = (0.8, 1.2)) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def _apply_saturation(
    img: Image.Image, factor_range: tuple = (0.8, 1.2)
) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Color(img).enhance(factor)


def _apply_blur(img: Image.Image, max_radius: float = 1.0) -> Image.Image:
    radius = random.uniform(0.0, max_radius)
    if radius < 0.1:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_noise(img: Image.Image, intensity: int = 8) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.uniform(-intensity, intensity, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _apply_gamma(img: Image.Image, gamma_range: tuple = (0.85, 1.15)) -> Image.Image:
    gamma = random.uniform(*gamma_range)
    arr = np.array(img).astype(np.float32) / 255.0
    corrected = np.clip(np.power(arr, gamma) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def _random_crop_and_resize(
    img: Image.Image, scale_range: tuple = (0.85, 1.0)
) -> Image.Image:
    width, height = img.size
    scale = random.uniform(*scale_range)
    crop_w = int(width * scale)
    crop_h = int(height * scale)
    left = random.randint(0, width - crop_w)
    top = random.randint(0, height - crop_h)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.BILINEAR)


def augment_image(img: Image.Image) -> Image.Image:
    """Aplica una secuencia aleatoria de transformaciones a una imagen."""
    transforms = [
        _apply_rotation,
        _apply_zoom,
        _apply_translation,
        _apply_brightness,
        _apply_contrast,
        _apply_saturation,
        _apply_blur,
        _apply_noise,
        _apply_gamma,
        _random_crop_and_resize,
    ]
    random.shuffle(transforms)

    for transform in transforms:
        if random.random() < 0.5:
            img = transform(img)

    return img


def discover_images(directory: Path) -> list[Path]:
    """Encuentra todas las imágenes soportadas en un directorio."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def generate_augmented_data(
    input_dir: str,
    output_dir: str,
    variants_per_image: int,
) -> None:
    """
    Genera imágenes aumentadas a partir de un dataset organizado por clases.

    Args:
        input_dir: Directorio con subcarpetas por clase.
        output_dir: Directorio de salida.
        variants_per_image: Número de variantes a generar por imagen original.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directorio de entrada no encontrado: {input_dir}")

    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No se encontraron subdirectorios de clases en {input_dir}")

    total_generated = 0
    for class_dir in class_dirs:
        images = discover_images(class_dir)
        if not images:
            logger.warning("No se encontraron imágenes en %s", class_dir)
            continue

        target_class_dir = output_path / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as exc:
                logger.error("No se pudo abrir %s: %s", image_path, exc)
                continue

            # Copiar imagen original
            original_target = target_class_dir / image_path.name
            img.save(original_target)

            # Generar variantes
            base_name = image_path.stem
            for i in range(variants_per_image):
                augmented = augment_image(img)
                output_file = target_class_dir / f"{base_name}_aug_{i:03d}.jpg"
                augmented.save(output_file, quality=95)
                total_generated += 1

        logger.info(
            "Clase %s: %d originales + %d variantes generadas",
            class_dir.name,
            len(images),
            len(images) * variants_per_image,
        )

    logger.info("Total de imágenes generadas: %d", total_generated)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generar datos aumentados a partir de imágenes de señas"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directorio de entrada organizado por clases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio de salida",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=20,
        help="Número de variantes a generar por imagen",
    )
    args = parser.parse_args()

    generate_augmented_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        variants_per_image=args.variants_per_image,
    )


if __name__ == "__main__":
    main()
