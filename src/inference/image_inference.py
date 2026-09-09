from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.inference.detector import ObjectDetector


def run_image_inference(
    image_path: Path,
    weights_path: Path,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    detector = ObjectDetector(weights_path)
    results = detector.predict_raw(image)

    annotated_image = results[0].plot()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(f"Failed to save: {output_path}")

    print(f"Saved result to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/inference_result.jpg"),
    )

    args = parser.parse_args()

    run_image_inference(
        image_path=args.image,
        weights_path=args.weights,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()