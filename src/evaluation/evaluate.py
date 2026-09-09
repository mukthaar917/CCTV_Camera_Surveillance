from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/merged_dataset/data.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/evaluation"),
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )

    names = getattr(metrics, "names", model.names)
    maps = getattr(metrics.box, "maps", None)

    per_class = []

    for class_id, class_name in names.items():
        value = (
            to_float(maps[class_id])
            if maps is not None and class_id < len(maps)
            else 0.0
        )

        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "map50_95": value,
            }
        )

    report = {
        "weights": str(args.weights.resolve()),
        "dataset": str(args.data.resolve()),
        "split": args.split,
        "device": device,
        "metrics": {
            "precision": to_float(metrics.box.mp),
            "recall": to_float(metrics.box.mr),
            "map50": to_float(metrics.box.map50),
            "map50_95": to_float(metrics.box.map),
        },
        "per_class": per_class,
        "results_directory": str(Path(metrics.save_dir).resolve()),
    }

    report_path = args.output / "evaluation_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL EVALUATION")
    print("=" * 80)
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Recall:    {report['metrics']['recall']:.4f}")
    print(f"mAP@50:    {report['metrics']['map50']:.4f}")
    print(f"mAP50-95:  {report['metrics']['map50_95']:.4f}")
    print(f"\nReport: {report_path.resolve()}")


if __name__ == "__main__":
    main()
