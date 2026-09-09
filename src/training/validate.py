from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from ultralytics import YOLO

from src.utils.logger import configure_logger


def validate_model(
    weights: Path,
    dataset_yaml: Path,
    image_size: int,
    batch_size: int,
) -> None:
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_yaml}"
        )

    device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=image_size,
        batch=batch_size,
        device=device,
        plots=True,
    )

    logger.info("mAP50: {:.4f}", metrics.box.map50)
    logger.info("mAP50-95: {:.4f}", metrics.box.map)
    logger.info("Precision: {:.4f}", metrics.box.mp)
    logger.info("Recall: {:.4f}", metrics.box.mr)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/merged_dataset/data.yaml"),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main() -> None:
    configure_logger()
    args = parse_arguments()

    validate_model(
        weights=args.weights,
        dataset_yaml=args.data,
        image_size=args.imgsz,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()