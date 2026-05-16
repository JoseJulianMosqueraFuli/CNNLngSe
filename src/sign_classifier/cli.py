import argparse
import logging
import sys

from tensorflow.keras.models import load_model

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    EPOCHS,
    BATCH_SIZE,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    MODEL_PATH,
    CLASSES,
    setup_logging,
)
from .train import train_model
from .predict import load_and_preprocess_image, predict_class

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    _, history = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_path=args.train_path,
        val_path=args.val_path,
        model_path=args.model_path,
        verbose=args.verbose,
    )
    best_acc = max(history.history["val_accuracy"])
    logger.info("Mejor accuracy de validación: %.4f", best_acc)


def predict(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    image = load_and_preprocess_image(args.image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    clase = predict_class(model, image, CLASSES)
    logger.info("Clase predicha: %s", clase)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clasificador de lenguaje de señas con CNN"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrenar el modelo")
    train_parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Número de épocas"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Tamaño del batch"
    )
    train_parser.add_argument(
        "--train-path",
        type=str,
        default=TRAINING_DATA_PATH,
        help="Ruta a datos de entrenamiento",
    )
    train_parser.add_argument(
        "--val-path",
        type=str,
        default=VALIDATION_DATA_PATH,
        help="Ruta a datos de validación",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Ruta donde guardar el modelo",
    )
    train_parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Nivel de verbosidad",
    )
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Predecir una imagen")
    predict_parser.add_argument(
        "image", type=str, help="Ruta a la imagen a clasificar"
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Ruta al modelo entrenado",
    )
    predict_parser.set_defaults(func=predict)

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
