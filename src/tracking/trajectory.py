from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import hypot
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    One recorded position of a tracked object.
    """

    x: int
    y: int
    frame_index: int
    timestamp_seconds: float


@dataclass(frozen=True)
class TrajectorySummary:
    """
    Movement summary for one tracked object.
    """

    track_id: int
    point_count: int
    start_point: tuple[int, int]
    current_point: tuple[int, int]
    displacement_pixels: float
    travelled_distance_pixels: float
    duration_seconds: float
    average_speed_pixels_per_second: float
    is_stationary: bool


class TrajectoryManager:
    """
    Stores and analyses object trajectories using ByteTrack IDs.

    Main responsibilities:
    - Save object centre points by track ID
    - Limit trajectory history length
    - Calculate displacement and travelled distance
    - Calculate approximate movement speed
    - Identify stationary objects
    - Draw trajectory paths on video frames
    - Remove inactive tracking histories
    """

    def __init__(
        self,
        max_points: int = 60,
        stationary_distance_threshold: float = 25.0,
        stationary_duration_threshold: float = 5.0,
        inactive_track_timeout: float = 10.0,
    ) -> None:
        if max_points < 2:
            raise ValueError("max_points must be at least 2.")

        if stationary_distance_threshold < 0:
            raise ValueError(
                "stationary_distance_threshold cannot be negative."
            )

        if stationary_duration_threshold < 0:
            raise ValueError(
                "stationary_duration_threshold cannot be negative."
            )

        if inactive_track_timeout < 0:
            raise ValueError(
                "inactive_track_timeout cannot be negative."
            )

        self.max_points = max_points
        self.stationary_distance_threshold = (
            stationary_distance_threshold
        )
        self.stationary_duration_threshold = (
            stationary_duration_threshold
        )
        self.inactive_track_timeout = inactive_track_timeout

        self._trajectories: dict[
            int,
            deque[TrajectoryPoint],
        ] = defaultdict(
            lambda: deque(maxlen=self.max_points)
        )

        self._last_seen: dict[int, float] = {}

    def update(
        self,
        track_id: int,
        center_x: float,
        center_y: float,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """
        Add a new trajectory point for one tracked object.
        """

        if track_id < 0:
            raise ValueError("track_id cannot be negative.")

        if frame_index < 0:
            raise ValueError("frame_index cannot be negative.")

        if timestamp_seconds < 0:
            raise ValueError(
                "timestamp_seconds cannot be negative."
            )

        point = TrajectoryPoint(
            x=int(round(center_x)),
            y=int(round(center_y)),
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

        trajectory = self._trajectories[track_id]

        # Avoid recording the exact same centre twice consecutively.
        if trajectory:
            previous = trajectory[-1]

            if (
                previous.x == point.x
                and previous.y == point.y
                and previous.frame_index == point.frame_index
            ):
                self._last_seen[track_id] = timestamp_seconds
                return

        trajectory.append(point)
        self._last_seen[track_id] = timestamp_seconds

    def update_from_detections(
        self,
        detections: Iterable[dict],
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """
        Update trajectories from pipeline detection dictionaries.

        Expected detection format:

        {
            "track_id": 5,
            "center": {
                "x": 320.5,
                "y": 210.2
            }
        }
        """

        for detection in detections:
            track_id = detection.get("track_id")
            center = detection.get("center")

            if track_id is None or not isinstance(center, dict):
                continue

            center_x = center.get("x")
            center_y = center.get("y")

            if center_x is None or center_y is None:
                continue

            try:
                self.update(
                    track_id=int(track_id),
                    center_x=float(center_x),
                    center_y=float(center_y),
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
            except (TypeError, ValueError):
                continue

    def get_points(
        self,
        track_id: int,
    ) -> list[TrajectoryPoint]:
        """
        Return the recorded trajectory points for one track ID.
        """

        return list(self._trajectories.get(track_id, []))

    def get_active_track_ids(self) -> list[int]:
        return sorted(self._trajectories.keys())

    def get_latest_point(
        self,
        track_id: int,
    ) -> TrajectoryPoint | None:
        trajectory = self._trajectories.get(track_id)

        if not trajectory:
            return None

        return trajectory[-1]

    def get_start_point(
        self,
        track_id: int,
    ) -> TrajectoryPoint | None:
        trajectory = self._trajectories.get(track_id)

        if not trajectory:
            return None

        return trajectory[0]

    def calculate_displacement(
        self,
        track_id: int,
    ) -> float:
        """
        Calculate straight-line distance between first and latest point.
        """

        trajectory = self._trajectories.get(track_id)

        if not trajectory or len(trajectory) < 2:
            return 0.0

        start = trajectory[0]
        end = trajectory[-1]

        return hypot(
            end.x - start.x,
            end.y - start.y,
        )

    def calculate_travelled_distance(
        self,
        track_id: int,
    ) -> float:
        """
        Calculate total travelled path length in pixels.
        """

        trajectory = self._trajectories.get(track_id)

        if not trajectory or len(trajectory) < 2:
            return 0.0

        distance = 0.0

        for previous, current in zip(
            trajectory,
            list(trajectory)[1:],
        ):
            distance += hypot(
                current.x - previous.x,
                current.y - previous.y,
            )

        return distance

    def calculate_duration(
        self,
        track_id: int,
    ) -> float:
        trajectory = self._trajectories.get(track_id)

        if not trajectory or len(trajectory) < 2:
            return 0.0

        return max(
            0.0,
            trajectory[-1].timestamp_seconds
            - trajectory[0].timestamp_seconds,
        )

    def calculate_average_speed(
        self,
        track_id: int,
    ) -> float:
        """
        Return approximate speed in pixels per second.
        """

        duration = self.calculate_duration(track_id)

        if duration <= 0:
            return 0.0

        travelled_distance = self.calculate_travelled_distance(
            track_id
        )

        return travelled_distance / duration

    def is_stationary(
        self,
        track_id: int,
    ) -> bool:
        """
        Determine whether an object has remained inside a small area
        for the configured minimum duration.

        This is useful for:
        - Package left near a door
        - Parked vehicle
        - Loitering pre-check
        - Abandoned object detection
        """

        duration = self.calculate_duration(track_id)

        if duration < self.stationary_duration_threshold:
            return False

        displacement = self.calculate_displacement(track_id)

        return displacement <= self.stationary_distance_threshold

    def get_summary(
        self,
        track_id: int,
    ) -> TrajectorySummary | None:
        trajectory = self._trajectories.get(track_id)

        if not trajectory:
            return None

        start = trajectory[0]
        current = trajectory[-1]

        displacement = self.calculate_displacement(track_id)
        travelled_distance = self.calculate_travelled_distance(
            track_id
        )
        duration = self.calculate_duration(track_id)
        average_speed = self.calculate_average_speed(track_id)

        return TrajectorySummary(
            track_id=track_id,
            point_count=len(trajectory),
            start_point=(start.x, start.y),
            current_point=(current.x, current.y),
            displacement_pixels=round(displacement, 3),
            travelled_distance_pixels=round(
                travelled_distance,
                3,
            ),
            duration_seconds=round(duration, 3),
            average_speed_pixels_per_second=round(
                average_speed,
                3,
            ),
            is_stationary=self.is_stationary(track_id),
        )

    def get_all_summaries(
        self,
    ) -> list[TrajectorySummary]:
        summaries: list[TrajectorySummary] = []

        for track_id in self.get_active_track_ids():
            summary = self.get_summary(track_id)

            if summary is not None:
                summaries.append(summary)

        return summaries

    def remove_inactive_tracks(
        self,
        current_timestamp_seconds: float,
    ) -> list[int]:
        """
        Remove tracks that have not appeared recently.

        Returns the IDs that were removed.
        """

        removed_track_ids: list[int] = []

        for track_id, last_seen in list(
            self._last_seen.items()
        ):
            inactive_duration = (
                current_timestamp_seconds - last_seen
            )

            if inactive_duration <= self.inactive_track_timeout:
                continue

            self._trajectories.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            removed_track_ids.append(track_id)

        return removed_track_ids

    def remove_track(
        self,
        track_id: int,
    ) -> None:
        self._trajectories.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def clear(self) -> None:
        self._trajectories.clear()
        self._last_seen.clear()

    def draw(
        self,
        frame: np.ndarray,
        show_track_id: bool = True,
        show_stationary_status: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw all active trajectories directly onto a frame.

        The frame is modified and returned.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Cannot draw on an empty frame.")

        for track_id, trajectory in self._trajectories.items():
            if len(trajectory) < 2:
                continue

            points = np.array(
                [(point.x, point.y) for point in trajectory],
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=(255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            latest = trajectory[-1]

            cv2.circle(
                frame,
                (latest.x, latest.y),
                radius=4,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            labels: list[str] = []

            if show_track_id:
                labels.append(f"ID {track_id}")

            if (
                show_stationary_status
                and self.is_stationary(track_id)
            ):
                labels.append("Stationary")

            if not labels:
                continue

            label = " | ".join(labels)

            cv2.putText(
                frame,
                label,
                (latest.x + 6, latest.y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def to_dict(
        self,
        track_id: int,
    ) -> dict | None:
        """
        Convert one trajectory summary to API-friendly JSON data.
        """

        summary = self.get_summary(track_id)

        if summary is None:
            return None

        points = self.get_points(track_id)

        return {
            "track_id": summary.track_id,
            "point_count": summary.point_count,
            "start_point": {
                "x": summary.start_point[0],
                "y": summary.start_point[1],
            },
            "current_point": {
                "x": summary.current_point[0],
                "y": summary.current_point[1],
            },
            "displacement_pixels": (
                summary.displacement_pixels
            ),
            "travelled_distance_pixels": (
                summary.travelled_distance_pixels
            ),
            "duration_seconds": summary.duration_seconds,
            "average_speed_pixels_per_second": (
                summary.average_speed_pixels_per_second
            ),
            "is_stationary": summary.is_stationary,
            "points": [
                {
                    "x": point.x,
                    "y": point.y,
                    "frame_index": point.frame_index,
                    "timestamp_seconds": (
                        point.timestamp_seconds
                    ),
                }
                for point in points
            ],
        }