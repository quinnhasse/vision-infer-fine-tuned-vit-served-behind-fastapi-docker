"""ViT-Small inference wrapper.

Loads a fine-tuned ViT checkpoint (or falls back to the hub model) and
exposes a single ``predict`` method that accepts a batch of PIL images and
returns per-image top-k class predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from src.config import settings

logger = logging.getLogger(__name__)


class VisionModel:
    """Thin wrapper around a HuggingFace ViT checkpoint.

    Parameters
    ----------
    model_name:
        HuggingFace hub model id or local directory.
    checkpoint:
        Optional path to a fine-tuned ``state_dict`` saved with
        ``torch.save(model.state_dict(), path)``.
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = settings.model_name,
        checkpoint: str = settings.model_checkpoint,
        device: str = settings.device,
    ) -> None:
        self.device = torch.device(device)
        logger.info("Loading processor from %s", model_name)
        self.processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(
            model_name
        )
        logger.info("Loading model from %s", model_name)
        self.model: ViTForImageClassification = (
            ViTForImageClassification.from_pretrained(model_name)
        )
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info("Loaded fine-tuned weights from %s", path)
        self.model.to(self.device)
        self.model.eval()
        self._label_map: dict[int, str] = self.model.config.id2label or {}
        logger.info("Model ready on %s", self.device)

    def predict(
        self, images: list[Image.Image], top_k: int = settings.top_k
    ) -> list[list[dict[str, Any]]]:
        """Run inference on a batch of PIL images.

        Parameters
        ----------
        images:
            List of RGB PIL images. They do not need to be pre-resized.
        top_k:
            Number of top predictions to return per image.

        Returns
        -------
        List of lists: one inner list per image, each containing dicts with
        ``label`` and ``score`` keys, sorted by score descending.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits: torch.Tensor = self.model(**inputs).logits  # (B, num_classes)

        probs = logits.softmax(dim=-1)  # (B, num_classes)
        k = min(top_k, probs.shape[-1])
        values, indices = probs.topk(k, dim=-1)  # (B, k)

        results: list[list[dict[str, Any]]] = []
        for img_values, img_indices in zip(values, indices):
            preds = [
                {
                    "label": self._label_map.get(idx.item(), str(idx.item())),
                    "score": round(float(val), 6),
                }
                for val, idx in zip(img_values, img_indices)
            ]
            results.append(preds)
        return results


# Module-level singleton; populated during server startup.
_model: VisionModel | None = None


def get_model() -> VisionModel:
    """Return the loaded model; raises if ``load_model`` was not called first."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() during startup.")
    return _model


def load_model() -> VisionModel:
    """Load the model into the module-level singleton and return it."""
    global _model
    _model = VisionModel()
    return _model
