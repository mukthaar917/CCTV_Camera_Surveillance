from pathlib import Path

import numpy as np
import pytest

from src.inference.detector import ObjectDetector


def test_missing_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ObjectDetector(tmp_path / "missing.pt")


def test_empty_frame(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"placeholder")

    class FakeYOLO:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr("src.inference.detector.YOLO", FakeYOLO)
    detector = ObjectDetector(weights)

    with pytest.raises(ValueError):
        detector.predict(np.array([], dtype=np.uint8))
