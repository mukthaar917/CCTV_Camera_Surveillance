from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetector:
    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}"
            )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device or (
            "0" if torch.cuda.is_available() else "cpu"
        )

        self.model = YOLO(str(self.weights_path))

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            names: dict[int, str] = result.names

            for box in result.boxes:
                coordinates = box.xyxy[0].cpu().tolist()
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=names[class_id],
                        confidence=confidence,
                        x1=float(coordinates[0]),
                        y1=float(coordinates[1]),
                        x2=float(coordinates[2]),
                        y2=float(coordinates[3]),
                    )
                )

        return detections

    def predict_raw(self, frame: np.ndarray) -> list[Any]:
        return self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )