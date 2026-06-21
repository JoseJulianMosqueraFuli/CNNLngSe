"""
Tests para la interfaz de línea de comandos.
"""

import pytest

from sign_classifier.cli import build_parser


def test_parser_has_train_command():
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.func is not None


def test_parser_has_predict_command():
    parser = build_parser()
    args = parser.parse_args(["predict", "imagen.jpg"])
    assert args.command == "predict"
    assert args.image == "imagen.jpg"


def test_parser_has_evaluate_command():
    parser = build_parser()
    args = parser.parse_args(["evaluate"])
    assert args.command == "evaluate"


def test_parser_has_batch_predict_command():
    parser = build_parser()
    args = parser.parse_args(["batch-predict", "./imagenes"])
    assert args.command == "batch-predict"
    assert args.input_dir == "./imagenes"


def test_parser_requires_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
