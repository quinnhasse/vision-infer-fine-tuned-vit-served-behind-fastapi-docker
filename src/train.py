"""Fine-tune ViT-Small on Food-101 using the HuggingFace Trainer + W&B.

Usage
-----
    python -m src.train \\
        --model_name WinKawaks/vit-small-patch16-224 \\
        --output_dir ./checkpoints/food101-vit-small \\
        --num_train_epochs 10 \\
        --per_device_train_batch_size 32 \\
        --learning_rate 2e-5 \\
        --warmup_ratio 0.1 \\
        --wandb_project vision-infer

Requirements
------------
Install training dependencies:

    pip install -r requirements-train.txt

Set WANDB_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial

import evaluate
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def build_transform(processor: ViTImageProcessor, is_train: bool):
    """Return a dataset-map function that preprocesses images.

    For training, we apply random horizontal flip and colour jitter before
    normalisation. For evaluation we only resize and normalise.
    """
    import torchvision.transforms as T

    if is_train:
        augment = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    else:
        augment = T.Compose([])

    def _transform(batch: dict) -> dict:
        images = [augment(img.convert("RGB")) for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    return _transform


def compute_metrics(eval_pred, metric) -> dict:
    """Compute accuracy from logits."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT on Food-101")
    parser.add_argument(
        "--model_name",
        default="WinKawaks/vit-small-patch16-224",
        help="HuggingFace hub model to start from",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints/food101-vit-small",
        help="Directory to write checkpoints and final model",
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_project", default="vision-infer")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load dataset from HuggingFace Hub.
    logger.info("Loading Food-101 dataset")
    dataset = load_dataset("food101", trust_remote_code=True)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    # Build label maps from the dataset.
    label_names: list[str] = train_ds.features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}
    num_labels = len(label_names)

    # Load processor and model.
    logger.info("Loading model %s", args.model_name)
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Preprocessor transforms.
    train_transform = build_transform(processor, is_train=True)
    val_transform = build_transform(processor, is_train=False)
    train_ds = train_ds.with_transform(train_transform)
    val_ds = val_ds.with_transform(val_transform)

    # Metric.
    accuracy = evaluate.load("accuracy")
    _compute_metrics = partial(compute_metrics, metric=accuracy)

    # Resume from checkpoint if one exists.
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            logger.info("Resuming from checkpoint %s", last_checkpoint)

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to=["wandb"],
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    # Train.
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Evaluate best checkpoint.
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)

    # Save final model + processor.
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    wandb.finish()
    logger.info("Done. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
