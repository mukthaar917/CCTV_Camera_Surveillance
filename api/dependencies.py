from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request, status
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.inference.detector import ObjectDetector
from src.utils.paths import PROJECT_ROOT, resolve_project_path


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Camera Surveillance AI API"
    app_version: str = "0.1.0"
    debug: bool = False

    model_weights_path: str = (
        "models/training_runs/baseline_yolo11n/weights/best.pt"
    )

    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )

    iou_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    image_size: int = Field(
        default=640,
        gt=0,
    )

    device: str | None = None

    max_upload_size_mb: int = Field(
        default=10,
        gt=0,
    )

    allowed_image_types: tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    )

    @property
    def resolved_weights_path(self) -> Path:
        return resolve_project_path(self.model_weights_path)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> APISettings:
    return APISettings()


def create_detector(
    settings: APISettings,
) -> ObjectDetector:
    weights_path = settings.resolved_weights_path

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights were not found: {weights_path}"
        )

    return ObjectDetector(
        weights_path=weights_path,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        image_size=settings.image_size,
        device=settings.device,
    )


def get_detector(
    request: Request,
) -> ObjectDetector:
    detector = getattr(
        request.app.state,
        "detector",
        None,
    )

    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model is not loaded.",
        )

    return detector


def get_model_path(
    request: Request,
) -> Path | None:
    model_path = getattr(
        request.app.state,
        "model_path",
        None,
    )

    if model_path is None:
        return None

    return Path(model_path)


def validate_content_type(
    content_type: str | None,
    settings: APISettings,
) -> None:
    if content_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file has no content type.",
        )

    normalized_content_type = (
        content_type
        .split(";", maxsplit=1)[0]
        .strip()
        .lower()
    )

    allowed_types = {
        item.lower()
        for item in settings.allowed_image_types
    }

    if normalized_content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image type. Allowed types: "
                + ", ".join(sorted(allowed_types))
            ),
        )


def validate_upload_size(
    file_size: int,
    settings: APISettings,
) -> None:
    if file_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if file_size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Uploaded file is too large. Maximum size is "
                f"{settings.max_upload_size_mb} MB."
            ),
        )