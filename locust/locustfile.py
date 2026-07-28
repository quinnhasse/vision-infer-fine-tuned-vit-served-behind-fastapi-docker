"""Locust load test for the vision-infer inference API.

Measures p50/p95/p99 latency under concurrent load.

Usage
-----
Run against a locally deployed server:

    locust -f locust/locustfile.py \\
        --host http://localhost:8000 \\
        --users 50 \\
        --spawn-rate 5 \\
        --run-time 60s \\
        --headless \\
        --csv results/load_test

Or open the Locust web UI (default port 8089):

    locust -f locust/locustfile.py --host http://localhost:8000

Environment variables
---------------------
API_KEY     Bearer token for authentication (default: changeme)
IMG_PATH    Path to a test image (default: uses a synthetic 64x64 PNG)
"""

from __future__ import annotations

import io
import os

from locust import HttpUser, between, task
from PIL import Image


def _make_png_bytes(width: int = 224, height: int = 224) -> bytes:
    """Generate a minimal RGB PNG in memory."""
    img = Image.new("RGB", (width, height), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Pre-generate the payload once; reuse across all virtual users.
_API_KEY = os.getenv("API_KEY", "changeme")
_IMG_PATH = os.getenv("IMG_PATH", "")

if _IMG_PATH and os.path.exists(_IMG_PATH):
    with open(_IMG_PATH, "rb") as fh:
        _IMAGE_BYTES = fh.read()
else:
    _IMAGE_BYTES = _make_png_bytes()


class InferenceUser(HttpUser):
    """Simulates a client posting images to the /predict endpoint."""

    wait_time = between(0.1, 0.5)  # seconds between requests per user

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {_API_KEY}"}

    @task(8)
    def predict_single(self) -> None:
        """POST a single image and record latency."""
        self.client.post(
            "/predict",
            files={"image": ("test.png", _IMAGE_BYTES, "image/png")},
            headers=self.headers,
            name="/predict (single)",
        )

    @task(2)
    def predict_batch(self) -> None:
        """POST a small batch (4 images) and record latency."""
        files = [
            ("images", ("img.png", _IMAGE_BYTES, "image/png")),
            ("images", ("img.png", _IMAGE_BYTES, "image/png")),
            ("images", ("img.png", _IMAGE_BYTES, "image/png")),
            ("images", ("img.png", _IMAGE_BYTES, "image/png")),
        ]
        self.client.post(
            "/predict/batch",
            files=files,
            headers=self.headers,
            name="/predict/batch (4-image)",
        )

    @task(1)
    def healthz(self) -> None:
        """Poll the health endpoint (no auth required)."""
        self.client.get("/healthz", name="/healthz")
