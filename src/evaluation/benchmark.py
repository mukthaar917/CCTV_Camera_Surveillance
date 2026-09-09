from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    return "0" if torch.cuda.is_available() else "cpu"


def synchronize(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0

    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/merged_dataset/valid/images"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/benchmark.json"),
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    images = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        raise RuntimeError(f"No benchmark images found: {args.images}")

    device = resolve_device(args.device)
    model = YOLO(str(args.weights))

    sample = cv2.imread(str(images[0]))

    if sample is None:
        raise RuntimeError(f"Unable to read: {images[0]}")

    for _ in range(args.warmup):
        model.predict(
            sample,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )

    synchronize(device)

    latencies: list[float] = []
    detection_counts: list[int] = []

    for _ in range(args.repetitions):
        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            synchronize(device)
            started_at = time.perf_counter()

            results = model.predict(
                image,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )

            synchronize(device)

            latency_ms = (time.perf_counter() - started_at) * 1000
            latencies.append(latency_ms)

            detection_counts.append(
                sum(
                    len(result.boxes)
                    for result in results
                    if result.boxes is not None
                )
            )

    if not latencies:
        raise RuntimeError("No benchmark inferences completed.")

    mean_latency = statistics.mean(latencies)

    report: dict[str, Any] = {
        "weights": str(args.weights.resolve()),
        "images_directory": str(args.images.resolve()),
        "device": device,
        "total_inferences": len(latencies),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "latency_ms": {
            "mean": round(mean_latency, 3),
            "median": round(statistics.median(latencies), 3),
            "minimum": round(min(latencies), 3),
            "maximum": round(max(latencies), 3),
            "p90": round(percentile(latencies, 0.90), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
        },
        "throughput_fps": round(1000 / mean_latency, 3),
        "average_detections": round(
            statistics.mean(detection_counts),
            3,
        ) if detection_counts else 0.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL BENCHMARK")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Inferences:   {report['total_inferences']}")
    print(f"Mean latency: {report['latency_ms']['mean']} ms")
    print(f"P95 latency:  {report['latency_ms']['p95']} ms")
    print(f"Estimated FPS:{report['throughput_fps']}")
    print(f"\nReport: {args.output.resolve()}")


if __name__ == "__main__":
    main()
