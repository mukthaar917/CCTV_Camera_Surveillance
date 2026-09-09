from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from api.dependencies import (
    APISettings,
    create_detector,
    get_detector,
    get_settings,
    validate_content_type,
    validate_upload_size,
)
from api.schemas import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.inference.detector import ObjectDetector


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    app.state.detector = None
    app.state.model_path = settings.resolved_weights_path

    if settings.resolved_weights_path.is_file():
        app.state.detector = create_detector(settings)

    yield

    app.state.detector = None


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    settings: APISettings = Depends(get_settings),
) -> HealthResponse:
    detector = getattr(
        app.state,
        "detector",
        None,
    )

    loaded = detector is not None

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_path=str(settings.resolved_weights_path),
        device=(
            detector.device
            if loaded
            else settings.device
        ),
    )


@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info(
    settings: APISettings = Depends(get_settings),
) -> ModelInfoResponse:
    detector = getattr(
        app.state,
        "detector",
        None,
    )

    if detector is None:
        return ModelInfoResponse(
            loaded=False,
            weights_path=str(settings.resolved_weights_path),
            device=settings.device,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            image_size=settings.image_size,
            classes={},
        )

    model_names = getattr(
        detector.model,
        "names",
        {},
    )

    classes = {
        int(class_id): str(class_name)
        for class_id, class_name in model_names.items()
    }

    return ModelInfoResponse(
        loaded=True,
        weights_path=str(detector.weights_path),
        device=str(detector.device),
        confidence_threshold=detector.confidence_threshold,
        iou_threshold=detector.iou_threshold,
        image_size=detector.image_size,
        classes=classes,
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
)
async def detect_objects(
    file: UploadFile = File(...),
    detector: ObjectDetector = Depends(get_detector),
    settings: APISettings = Depends(get_settings),
) -> DetectionResponse:
    validate_content_type(
        content_type=file.content_type,
        settings=settings,
    )

    contents = await file.read()

    validate_upload_size(
        file_size=len(contents),
        settings=settings,
    )

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image.",
        )

    started_at = time.perf_counter()

    detections = detector.predict(frame)

    inference_time_ms = (
        time.perf_counter() - started_at
    ) * 1000.0

    image_height, image_width = frame.shape[:2]

    return DetectionResponse(
        count=len(detections),
        image_width=image_width,
        image_height=image_height,
        inference_time_ms=round(
            inference_time_ms,
            3,
        ),
        detections=[
            DetectionItem(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=round(
                    detection.confidence,
                    4,
                ),
                bounding_box=BoundingBox(
                    x1=round(detection.x1, 2),
                    y1=round(detection.y1, 2),
                    x2=round(detection.x2, 2),
                    y2=round(detection.y2, 2),
                ),
            )
            for detection in detections
        ],
    )