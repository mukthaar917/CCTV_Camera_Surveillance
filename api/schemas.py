from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float = Field(
        ...,
        description="Left bounding-box coordinate in pixels.",
    )
    y1: float = Field(
        ...,
        description="Top bounding-box coordinate in pixels.",
    )
    x2: float = Field(
        ...,
        description="Right bounding-box coordinate in pixels.",
    )
    y2: float = Field(
        ...,
        description="Bottom bounding-box coordinate in pixels.",
    )


class DetectionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(
        ...,
        ge=0,
        description="Numeric class identifier produced by YOLO.",
    )
    class_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable detected class name.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence between 0 and 1.",
    )
    bounding_box: BoundingBox


class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(
        ...,
        ge=0,
        description="Number of detections returned.",
    )
    image_width: int = Field(
        ...,
        gt=0,
        description="Input image width in pixels.",
    )
    image_height: int = Field(
        ...,
        gt=0,
        description="Input image height in pixels.",
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total model inference time in milliseconds.",
    )
    detections: list[DetectionItem]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_path: str | None = None
    device: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loaded: bool
    weights_path: str | None = None
    device: str | None = None
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    image_size: int | None = None
    classes: dict[int, str] = Field(default_factory=dict)