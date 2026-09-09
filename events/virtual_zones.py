from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import cv2
import numpy as np


class ZoneType(str, Enum):
    RESTRICTED = "restricted"
    ENTRY_EXIT = "entry_exit"
    MONITORING = "monitoring"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class VirtualZone:
    zone_id: str
    name: str
    points: list[Point]
    zone_type: ZoneType = ZoneType.MONITORING
    normalized: bool = False
    enabled: bool = True
    allowed_class_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.zone_id.strip() or not self.name.strip():
            raise ValueError("Zone ID and name cannot be empty.")
        if len(self.points) < 3:
            raise ValueError("A polygon zone requires at least three points.")
        if self.normalized:
            for point in self.points:
                if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                    raise ValueError("Normalized coordinates must be within 0..1.")

    def pixel_points(self, frame_width: int, frame_height: int) -> np.ndarray:
        converted = []
        for point in self.points:
            x = point.x * frame_width if self.normalized else point.x
            y = point.y * frame_height if self.normalized else point.y
            converted.append((int(round(x)), int(round(y))))
        return np.array(converted, dtype=np.int32)

    def contains_point(self, x: float, y: float, frame_width: int, frame_height: int) -> bool:
        polygon = self.pixel_points(frame_width, frame_height)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def accepts_class(self, class_id: int) -> bool:
        return not self.allowed_class_ids or class_id in self.allowed_class_ids

    def draw(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        polygon = self.pixel_points(width, height)
        cv2.polylines(frame, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
        x, y = polygon[0]
        cv2.putText(frame, self.name, (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


@dataclass(frozen=True)
class ZoneObservation:
    zone_id: str
    zone_name: str
    track_id: int
    class_id: int
    class_name: str
    is_inside: bool
    center_x: float
    center_y: float


class VirtualZoneEngine:
    def __init__(self, zones: Iterable[VirtualZone] | None = None) -> None:
        self._zones: dict[str, VirtualZone] = {}
        for zone in zones or []:
            self.upsert_zone(zone)

    def upsert_zone(self, zone: VirtualZone) -> None:
        self._zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> None:
        self._zones.pop(zone_id, None)

    def list_zones(self) -> list[VirtualZone]:
        return list(self._zones.values())

    def evaluate_detections(self, detections: Iterable[dict], frame_width: int, frame_height: int) -> list[ZoneObservation]:
        observations: list[ZoneObservation] = []
        for detection in detections:
            track_id = detection.get("track_id")
            class_id = detection.get("class_id")
            center = detection.get("center")
            if track_id is None or class_id is None or not isinstance(center, dict):
                continue
            if center.get("x") is None or center.get("y") is None:
                continue
            for zone in self._zones.values():
                if not zone.enabled or not zone.accepts_class(int(class_id)):
                    continue
                observations.append(ZoneObservation(
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=str(detection.get("class_name", "unknown")),
                    is_inside=zone.contains_point(float(center["x"]), float(center["y"]), frame_width, frame_height),
                    center_x=float(center["x"]),
                    center_y=float(center["y"]),
                ))
        return observations

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        for zone in self._zones.values():
            if zone.enabled:
                zone.draw(frame)
        return frame
