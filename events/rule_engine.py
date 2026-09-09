from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from events.virtual_zones import ZoneObservation


class EventType(str, Enum):
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    VEHICLE_ENTRY = "vehicle_entry"
    ANIMAL_INTRUSION = "animal_intrusion"
    PACKAGE_STATIONARY = "package_stationary"
    FIRE_DETECTED = "fire_detected"
    SMOKE_DETECTED = "smoke_detected"


@dataclass(frozen=True)
class SecurityEvent:
    event_type: EventType
    message: str
    timestamp: str
    track_id: int | None = None
    class_id: int | None = None
    class_name: str | None = None
    zone_id: str | None = None
    zone_name: str | None = None
    confidence: float | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data


class RuleEngine:
    def __init__(self, loitering_seconds: float = 90.0, cooldown_seconds: float = 10.0) -> None:
        self.loitering_seconds = loitering_seconds
        self.cooldown_seconds = cooldown_seconds
        self._inside: dict[tuple[str, int], bool] = {}
        self._entered_at: dict[tuple[str, int], float] = {}
        self._last_event: dict[tuple[str, str, int | None], float] = {}

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _can_emit(self, event_type: EventType, zone_id: str, track_id: int | None, now: float) -> bool:
        key = (event_type.value, zone_id, track_id)
        previous = self._last_event.get(key)
        if previous is not None and now - previous < self.cooldown_seconds:
            return False
        self._last_event[key] = now
        return True

    def process_zone_observations(self, observations: Iterable[ZoneObservation], timestamp_seconds: float) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for observation in observations:
            key = (observation.zone_id, observation.track_id)
            was_inside = self._inside.get(key, False)
            is_inside = observation.is_inside

            if is_inside and not was_inside:
                self._entered_at[key] = timestamp_seconds
                if self._can_emit(EventType.ZONE_ENTRY, observation.zone_id, observation.track_id, timestamp_seconds):
                    events.append(SecurityEvent(EventType.ZONE_ENTRY, f"{observation.class_name.title()} entered {observation.zone_name}.", self._timestamp(), observation.track_id, observation.class_id, observation.class_name, observation.zone_id, observation.zone_name))

                event_type = {1: EventType.INTRUSION, 0: EventType.VEHICLE_ENTRY, 4: EventType.ANIMAL_INTRUSION}.get(observation.class_id)
                if event_type and self._can_emit(event_type, observation.zone_id, observation.track_id, timestamp_seconds):
                    events.append(SecurityEvent(event_type, f"{observation.class_name.title()} detected inside {observation.zone_name}.", self._timestamp(), observation.track_id, observation.class_id, observation.class_name, observation.zone_id, observation.zone_name))

            elif not is_inside and was_inside:
                self._entered_at.pop(key, None)
                if self._can_emit(EventType.ZONE_EXIT, observation.zone_id, observation.track_id, timestamp_seconds):
                    events.append(SecurityEvent(EventType.ZONE_EXIT, f"{observation.class_name.title()} exited {observation.zone_name}.", self._timestamp(), observation.track_id, observation.class_id, observation.class_name, observation.zone_id, observation.zone_name))

            elif is_inside and was_inside and observation.class_id == 1:
                duration = timestamp_seconds - self._entered_at.get(key, timestamp_seconds)
                if duration >= self.loitering_seconds and self._can_emit(EventType.LOITERING, observation.zone_id, observation.track_id, timestamp_seconds):
                    events.append(SecurityEvent(EventType.LOITERING, f"Person remained in {observation.zone_name} for {duration:.1f} seconds.", self._timestamp(), observation.track_id, observation.class_id, observation.class_name, observation.zone_id, observation.zone_name, metadata={"duration_seconds": round(duration, 2)}))

            self._inside[key] = is_inside
        return events

    def process_detection_events(self, detections: Iterable[dict], timestamp_seconds: float) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for detection in detections:
            class_id = detection.get("class_id")
            mapping = {2: (EventType.FIRE_DETECTED, "Fire detected in the camera view."), 3: (EventType.SMOKE_DETECTED, "Smoke detected in the camera view.")}
            if class_id not in mapping:
                continue
            event_type, message = mapping[class_id]
            track_id = detection.get("track_id")
            if self._can_emit(event_type, "global", track_id, timestamp_seconds):
                events.append(SecurityEvent(event_type, message, self._timestamp(), track_id, class_id, detection.get("class_name"), confidence=detection.get("confidence")))
        return events

    def reset(self) -> None:
        self._inside.clear()
        self._entered_at.clear()
        self._last_event.clear()
