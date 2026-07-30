"""Pytest fixtures shared across all test modules."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Fake model fixture
# ---------------------------------------------------------------------------

FAKE_PREDICTIONS = [
    [
        {"label": "pizza", "score": 0.912345},
        {"label": "hamburger", "score": 0.054321},
        {"label": "hot_dog", "score": 0.012345},
        {"label": "french_fries", "score": 0.009876},
        {"label": "sushi", "score": 0.004321},
    ]
]


def _make_fake_model() -> MagicMock:
    model = MagicMock()
    model.predict.side_effect = lambda images, top_k=5: [
        FAKE_PREDICTIONS[0][:top_k] for _ in images
    ]
    return model


@pytest.fixture()
def fake_model() -> MagicMock:
    return _make_fake_model()


# ---------------------------------------------------------------------------
# Test client with mocked model
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(fake_model) -> TestClient:
    """Return a TestClient with the model singleton replaced by a mock."""
    import src.server as server_module

    # Patch load_model so lifespan does not download weights.
    with patch("src.server.load_model", return_value=fake_model):
        # Also reset the batch queue singleton between tests.
        server_module._batch_queue = None
        with patch("src.model._model", fake_model):
            with TestClient(server_module.app, raise_server_exceptions=True) as c:
                # Override the queue's model reference.
                server_module._batch_queue = None
                with patch(
                    "src.server.get_batch_queue",
                    wraps=lambda: _patched_queue(fake_model),
                ):
                    yield c


def _patched_queue(model: MagicMock):
    """Build a BatchQueue wired to the fake model."""
    from src.server import BatchQueue

    q = BatchQueue(model=model, timeout_ms=10, max_size=32)
    return q


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def png_bytes(width: int = 64, height: int = 64, color=(200, 100, 50)) -> bytes:
    """Return a minimal RGB PNG as bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_png() -> bytes:
    return png_bytes()
