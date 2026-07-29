"""Export a fine-tuned ViT checkpoint to ONNX.

The exported graph accepts a ``pixel_values`` input of shape
``(batch, 3, height, width)`` and produces ``logits`` of shape
``(batch, num_classes)``.

Usage
-----
    python -m src.export_onnx \\
        --checkpoint ./checkpoints/food101-vit-small \\
        --output ./checkpoints/food101-vit-small.onnx \\
        --opset 17

The ``--checkpoint`` flag can point to:
  - A local directory produced by ``train.py`` (contains ``config.json``
    and model weights).
  - A HuggingFace hub model id (falls back to hub download).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ViT to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="google/vit-base-patch16-224",
        help="Local checkpoint directory or HuggingFace hub model id",
    )
    parser.add_argument(
        "--output",
        default="checkpoint.onnx",
        help="Output path for the ONNX file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used for the dummy input during tracing",
    )
    return parser.parse_args()


def export(
    checkpoint: str,
    output: str,
    opset: int = 17,
    batch_size: int = 1,
) -> Path:
    """Load ``checkpoint``, trace through a dummy input, and write ONNX.

    Parameters
    ----------
    checkpoint:
        HuggingFace hub id or local directory.
    output:
        Destination ``.onnx`` path.
    opset:
        ONNX opset version. 17 is the minimum recommended for ViT.
    batch_size:
        Batch size for the tracing dummy input.

    Returns
    -------
    Path to the written ONNX file.
    """
    logger.info("Loading processor from %s", checkpoint)
    processor = ViTImageProcessor.from_pretrained(checkpoint)
    image_size = processor.size.get("height", 224)

    logger.info("Loading model from %s", checkpoint)
    model = ViTForImageClassification.from_pretrained(checkpoint)
    model.eval()

    dummy_input = torch.zeros(
        (batch_size, 3, image_size, image_size), dtype=torch.float32
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to %s (opset %d)", out_path, opset)
    torch.onnx.export(
        model,
        ({"pixel_values": dummy_input},),
        str(out_path),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Exported %.1f MB ONNX model to %s", size_mb, out_path)
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    export(
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
