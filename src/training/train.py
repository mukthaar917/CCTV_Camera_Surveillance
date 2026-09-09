from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from ultralytics import YOLO

from src.utils.config import load_yaml_config
from src.utils.logger import configure_logger


def resolve_device(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device

    return "0" if torch.cuda.is_available() else "cpu"


def train_model(config_path: str | Path) -> None:
    config: dict[str, Any] = load_yaml_config(config_path)

    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    dataset_yaml = Path(dataset_config["yaml_path"]).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML was not found: {dataset_yaml}"
        )

    model_name = model_config["pretrained_weights"]
    device = resolve_device(str(training_config["device"]))

    logger.info("Loading model: {}", model_name)
    logger.info("Dataset YAML: {}", dataset_yaml)
    logger.info("Training device: {}", device)

    model = YOLO(model_name)

    results = model.train(
        data=str(dataset_yaml),
        epochs=int(training_config["epochs"]),
        imgsz=int(training_config["image_size"]),
        batch=int(training_config["batch_size"]),
        device=device,
        workers=int(training_config["workers"]),
        patience=int(training_config["patience"]),
        optimizer=str(training_config["optimizer"]),
        seed=int(training_config["seed"]),
        deterministic=bool(training_config["deterministic"]),
        cache=bool(training_config["cache"]),
        pretrained=bool(training_config["pretrained"]),
        close_mosaic=int(training_config["close_mosaic"]),
        project=str(output_config["project_dir"]),
        name=str(output_config["run_name"]),
        save_period=int(output_config["save_period"]),
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    logger.success("Training completed.")
    logger.info("Results directory: {}", results.save_dir)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Camera Surveillance YOLO model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training.yaml"),
        help="Path to the training YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logger()
    arguments = parse_arguments()
    train_model(arguments.config)


if __name__ == "__main__":
    main()