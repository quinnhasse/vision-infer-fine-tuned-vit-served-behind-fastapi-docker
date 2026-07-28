"""FastAPI inference server.

Endpoints
---------
POST /predict
    Single-image inference. Accepts multipart/form-data with an ``image`` field.

POST /predict/batch
    Batch inference. Accepts multipart/form-data with one or more ``images`` fields.

GET /healthz
    Liveness check — returns 200 if the model is loaded.

Authentication
--------------
All /predict endpoints require ``Authorization: Bearer <key>`` where the key
matches the ``API_KEY`` environment variable.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel

from src.config import settings
from src.model import VisionModel, load_model

logger = logging.getLogger(__name__)
security = HTTPBearer()


# ---------------------------------------------------------------------------
# Lifespan — warm model load at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the server accepts traffic."""
    logger.info("Loading model during startup")
    app.state.model = load_model()
    logger.info("Model loaded — server ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="vision-infer",
    description="ViT image classification inference API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Validate the Bearer token against API_KEY."""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: list[list[Prediction]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Batch queue
# ---------------------------------------------------------------------------


class _BatchItem:
    """One pending inference request."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.future: asyncio.Future[list[dict[str, Any]]] = asyncio.get_event_loop().create_future()


class BatchQueue:
    """Accumulates single-image requests and flushes them as a batch.

    Images are collected for up to ``timeout_ms`` milliseconds or until
    ``max_size`` is reached, then processed together.
    """

    def __init__(
        self,
        model: VisionModel,
        timeout_ms: int = settings.batch_timeout_ms,
        max_size: int = settings.max_batch_size,
    ) -> None:
        self._model = model
        self._timeout = timeout_ms / 1000.0
        self._max_size = max_size
        self._queue: list[_BatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, image: Image.Image) -> list[dict[str, Any]]:
        """Add an image to the queue and wait for its prediction."""
        item = _BatchItem(image)
        async with self._lock:
            self._queue.append(item)
            if len(self._queue) >= self._max_size:
                await self._flush()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._deferred_flush())
        return await item.future

    async def _deferred_flush(self) -> None:
        await asyncio.sleep(self._timeout)
        async with self._lock:
            if self._queue:
                await self._flush()

    async def _flush(self) -> None:
        """Process the current queue contents; must be called under _lock."""
        batch = self._queue[:]
        self._queue.clear()
        images = [item.image for item in batch]
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: self._model.predict(images, top_k=settings.top_k)
            )
            for item, preds in zip(batch, results):
                item.future.set_result(preds)
        except Exception as exc:  # noqa: BLE001
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)


# Lazy singleton; initialised after the model is ready.
_batch_queue: BatchQueue | None = None


def get_batch_queue() -> BatchQueue:
    global _batch_queue
    if _batch_queue is None:
        from src.model import get_model

        _batch_queue = BatchQueue(model=get_model())
    return _batch_queue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_upload(upload: UploadFile) -> Image.Image:
    """Read an uploaded file and return a PIL RGB image."""
    raw = upload.file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        ) from exc
    return img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    """Classify a single image.

    Returns the top-k label/score pairs ranked by confidence.
    Requests are batched internally; individual latency reflects queue wait.
    """
    img = _decode_upload(image)
    queue = get_batch_queue()
    t0 = time.perf_counter()
    preds = await queue.submit(img)
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=[Prediction(**p) for p in preds],
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch(
    images: list[UploadFile] = File(...),
) -> BatchPredictResponse:
    """Classify multiple images in one request.

    Each image is added to the shared batch queue independently, so this
    endpoint benefits from the same internal batching as single requests.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one image is required",
        )
    if len(images) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}",
        )
    pil_images = [_decode_upload(img) for img in images]
    queue = get_batch_queue()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[queue.submit(img) for img in pil_images])
    latency_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        results=[[Prediction(**p) for p in preds] for preds in results],
        latency_ms=round(latency_ms, 2),
    )
