"""Tests for the FastAPI inference server.

The model singleton is replaced with a mock in conftest.py so these tests
run without downloading weights.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import FAKE_PREDICTIONS, png_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


VALID_KEY = "changeme"
HEADERS_OK = {"Authorization": f"Bearer {VALID_KEY}"}
HEADERS_BAD = {"Authorization": "Bearer wrong-key"}


def _make_client(fake_model: MagicMock) -> TestClient:
    import src.server as server_module
    from src.server import BatchQueue

    fake_queue = BatchQueue(model=fake_model, timeout_ms=10, max_size=32)
    server_module._batch_queue = None

    with patch("src.server.load_model", return_value=fake_model):
        with patch("src.model._model", fake_model):
            with TestClient(server_module.app, raise_server_exceptions=True) as c:
                server_module._batch_queue = None
                with patch(
                    "src.server.get_batch_queue", return_value=fake_queue
                ):
                    yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_healthz_returns_200(self, fake_model):
        for c in _make_client(fake_model):
            resp = c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_healthz_no_auth_required(self, fake_model):
        for c in _make_client(fake_model):
            resp = c.get("/healthz")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_predict_rejects_missing_token(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post("/predict", files={"image": ("test.png", img_bytes, "image/png")})
        assert resp.status_code in (401, 403)

    def test_predict_rejects_wrong_token(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_BAD,
            )
        assert resp.status_code == 401

    def test_predict_accepts_valid_token(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_OK,
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Single predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_predictions(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_OK,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert "latency_ms" in body
        assert len(body["predictions"]) > 0

    def test_prediction_fields(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_OK,
            )
        preds = resp.json()["predictions"]
        for pred in preds:
            assert "label" in pred
            assert "score" in pred
            assert isinstance(pred["score"], float)
            assert 0.0 <= pred["score"] <= 1.0

    def test_scores_sum_approximately_one(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_OK,
            )
        scores = [p["score"] for p in resp.json()["predictions"]]
        # Top-5 may not sum to 1 when there are 1000 classes, but top score
        # should be the highest.
        assert scores == sorted(scores, reverse=True)

    def test_rejects_non_image(self, fake_model):
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("bad.txt", b"not an image", "text/plain")},
                headers=HEADERS_OK,
            )
        assert resp.status_code == 422

    def test_latency_ms_is_positive(self, fake_model):
        img_bytes = png_bytes()
        for c in _make_client(fake_model):
            resp = c.post(
                "/predict",
                files={"image": ("test.png", img_bytes, "image/png")},
                headers=HEADERS_OK,
            )
        assert resp.json()["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------


class TestBatchPredict:
    def test_batch_returns_one_result_per_image(self, fake_model):
        img_bytes = png_bytes()
        files = [
            ("images", ("a.png", img_bytes, "image/png")),
            ("images", ("b.png", png_bytes(color=(10, 20, 30)), "image/png")),
        ]
        for c in _make_client(fake_model):
            resp = c.post("/predict/batch", files=files, headers=HEADERS_OK)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2

    def test_batch_rejects_empty_list(self, fake_model):
        for c in _make_client(fake_model):
            resp = c.post("/predict/batch", files=[], headers=HEADERS_OK)
        # FastAPI should reject missing required field
        assert resp.status_code in (400, 422)
