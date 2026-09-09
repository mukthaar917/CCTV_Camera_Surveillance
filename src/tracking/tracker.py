from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class ByteTrackTracker:
    def __init__(
        self,
        weights_path: str | Path,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.35,
        image_size: int = 640,
    ) -> None:
        self.model = YOLO(str(weights_path))
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

    def track_frame(self, frame: np.ndarray) -> list[Any]:
        return self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )