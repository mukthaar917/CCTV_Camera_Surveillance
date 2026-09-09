from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, source: str | int | Path) -> None:
        self.source = str(source) if isinstance(source, Path) else source
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

    @property
    def fps(self) -> float:
        value = self.capture.get(cv2.CAP_PROP_FPS)
        return value if value > 0 else 30.0

    @property
    def frame_width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            success, frame = self.capture.read()

            if not success:
                break

            yield frame

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()