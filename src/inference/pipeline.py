from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.inference.video_processor import VideoProcessor
from src.utils.logger import configure_logger


class SurveillancePipeline:
    """
    End-to-end surveillance inference pipeline:

        Video frame
            -> YOLO detection
            -> ByteTrack tracking
            -> annotated frame
            -> optional output video

    This is the first baseline pipeline. Event rules and virtual zones
    can be connected later through the `process_events()` method.
    """

    def __init__(
        self,
        weights_path: str | Path,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        image_size: int = 640,
        device: str | int | None = None,
        trajectory_length: int = 30,
    ) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        self.model = YOLO(str(self.weights_path))
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.trajectory_length = trajectory_length

        self.trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trajectory_length)
        )

        self.total_frames = 0
        self.total_processing_time = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]], float]:
        """
        Run YOLO and ByteTrack on one frame.

        Returns:
            annotated_frame
            detections
            inference_time_ms
        """

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        started_at = time.perf_counter()

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        self.total_frames += 1
        self.total_processing_time += inference_time_ms / 1000.0

        if not results:
            return frame.copy(), [], inference_time_ms

        result = results[0]
        annotated_frame = result.plot()
        detections = self._extract_detections(result)

        self._update_trajectories(detections)
        self._draw_trajectories(annotated_frame)
        self._draw_performance_overlay(
            frame=annotated_frame,
            inference_time_ms=inference_time_ms,
            detection_count=len(detections),
        )

        return annotated_frame, detections, inference_time_ms

    def _extract_detections(
        self,
        result: Any,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for index in range(len(boxes)):
            box = boxes[index]

            coordinates = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            track_id: int | None = None

            if box.id is not None:
                track_id = int(box.id[0].item())

            x1, y1, x2, y2 = map(float, coordinates)

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": str(result.names[class_id]),
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                    "center": {
                        "x": round((x1 + x2) / 2.0, 2),
                        "y": round((y1 + y2) / 2.0, 2),
                    },
                }
            )

        return detections

    def _update_trajectories(
        self,
        detections: list[dict[str, Any]],
    ) -> None:
        for detection in detections:
            track_id = detection["track_id"]

            if track_id is None:
                continue

            center = detection["center"]

            point = (
                int(center["x"]),
                int(center["y"]),
            )

            self.trajectories[track_id].append(point)

    def _draw_trajectories(
        self,
        frame: np.ndarray,
    ) -> None:
        for track_id, points in self.trajectories.items():
            if len(points) < 2:
                continue

            points_array = np.array(
                points,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [points_array],
                isClosed=False,
                color=(255, 255, 255),
                thickness=2,
            )

            latest_point = points[-1]

            cv2.putText(
                frame,
                f"ID {track_id}",
                (
                    latest_point[0] + 5,
                    latest_point[1] - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_performance_overlay(
        self,
        frame: np.ndarray,
        inference_time_ms: float,
        detection_count: int,
    ) -> None:
        current_fps = (
            1000.0 / inference_time_ms
            if inference_time_ms > 0
            else 0.0
        )

        average_fps = (
            self.total_frames / self.total_processing_time
            if self.total_processing_time > 0
            else 0.0
        )

        overlay_lines = [
            f"Detections: {detection_count}",
            f"Inference: {inference_time_ms:.1f} ms",
            f"Current FPS: {current_fps:.1f}",
            f"Average FPS: {average_fps:.1f}",
        ]

        overlay_height = 30 + len(overlay_lines) * 25

        cv2.rectangle(
            frame,
            (10, 10),
            (260, overlay_height),
            (0, 0, 0),
            thickness=-1,
        )

        for index, text in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text,
                (20, 38 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def process_events(
        self,
        detections: list[dict[str, Any]],
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[dict[str, Any]]:
        """
        Placeholder for Krishna's event and virtual-zone engine.

        Later, pass detections to:
            events/virtual_zones.py
            events/rule_engine.py
        """

        return []

    def run(
        self,
        source: str | int | Path,
        output_path: str | Path | None = None,
        display: bool = True,
    ) -> None:
        """
        Process a video file, RTSP URL, or webcam.

        Examples:
            source=0
            source="sample.mp4"
            source="rtsp://username:password@camera-ip/stream"
        """

        output_writer: cv2.VideoWriter | None = None
        frame_index = 0

        with VideoProcessor(source) as processor:
            source_fps = processor.fps

            logger.info("Video source opened: {}", source)
            logger.info(
                "Resolution: {}x{}",
                processor.frame_width,
                processor.frame_height,
            )
            logger.info("Source FPS: {:.2f}", source_fps)

            if output_path is not None:
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)

                codec = cv2.VideoWriter_fourcc(*"mp4v")

                output_writer = cv2.VideoWriter(
                    str(output),
                    codec,
                    source_fps,
                    (
                        processor.frame_width,
                        processor.frame_height,
                    ),
                )

                if not output_writer.isOpened():
                    raise RuntimeError(
                        f"Unable to create output video: {output}"
                    )

            try:
                for frame in processor.frames():
                    frame_index += 1

                    annotated_frame, detections, _ = self.process_frame(
                        frame
                    )

                    timestamp_seconds = frame_index / source_fps

                    self.process_events(
                        detections=detections,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                    )

                    if output_writer is not None:
                        output_writer.write(annotated_frame)

                    if display:
                        cv2.imshow(
                            "Camera Surveillance AI",
                            annotated_frame,
                        )

                        key = cv2.waitKey(1) & 0xFF

                        if key in (ord("q"), 27):
                            logger.info(
                                "Processing stopped by the user."
                            )
                            break

            finally:
                if output_writer is not None:
                    output_writer.release()

                if display:
                    cv2.destroyAllWindows()

        logger.success(
            "Video processing completed. Frames processed: {}",
            frame_index,
        )

    def get_statistics(self) -> dict[str, float | int]:
        average_fps = (
            self.total_frames / self.total_processing_time
            if self.total_processing_time > 0
            else 0.0
        )

        average_processing_time_ms = (
            self.total_processing_time
            / self.total_frames
            * 1000.0
            if self.total_frames > 0
            else 0.0
        )

        return {
            "total_frames": self.total_frames,
            "total_processing_time_seconds": round(
                self.total_processing_time,
                3,
            ),
            "average_processing_time_ms": round(
                average_processing_time_ms,
                3,
            ),
            "average_fps": round(average_fps, 3),
            "active_trajectories": len(self.trajectories),
        }


def parse_source(value: str) -> str | int:
    """
    Convert webcam numbers such as '0' into integers.
    Keep file paths and RTSP URLs as strings.
    """

    stripped = value.strip()

    if stripped.isdigit():
        return int(stripped)

    return stripped


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO and ByteTrack on a video, webcam, or RTSP stream."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to best.pt or another YOLO weight file.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path, webcam number, or RTSP URL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the annotated MP4 video.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device such as 0, cpu, or cuda:0.",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening an OpenCV window.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logger()
    args = parse_arguments()

    pipeline = SurveillancePipeline(
        weights_path=args.weights,
        tracker_config=args.tracker,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=args.device,
    )

    pipeline.run(
        source=parse_source(args.source),
        output_path=args.output,
        display=not args.no_display,
    )

    logger.info(
        "Pipeline statistics: {}",
        pipeline.get_statistics(),
    )


if __name__ == "__main__":
    main()