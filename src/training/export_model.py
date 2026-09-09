from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_model(
    weights_path: Path,
    export_format: str,
    image_size: int,
) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}"
        )

    model = YOLO(str(weights_path))

    exported_path = model.export(
        format=export_format,
        imgsz=image_size,
        simplify=True,
        dynamic=False,
    )

    print(f"Exported model: {exported_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--format",
        choices=["onnx", "engine", "torchscript"],
        default="onnx",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    export_model(
        weights_path=args.weights,
        export_format=args.format,
        image_size=args.imgsz,
    )


if __name__ == "__main__":
    main()