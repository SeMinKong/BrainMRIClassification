import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import CONFIG
from src.data_loader import get_datasets

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Render and save a confusion-matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info("Confusion matrix saved to: %s", save_path)


def main(ckpt_path: str, cfg: dict = CONFIG) -> None:
    train_ds, val_ds = get_datasets(cfg)

    # Retrieve class names from a fresh (unprocessed) dataset object
    try:
        original_ds = tf.keras.utils.image_dataset_from_directory(
            cfg["dataset_dir"],
            validation_split=cfg["split"]["val"],
            subset="training",
            seed=cfg["seed"],
            image_size=(cfg["img_height"], cfg["img_width"]),
            batch_size=cfg["batch_size"],
        )
    except FileNotFoundError as exc:
        logger.error("Dataset directory not found: %s", cfg["dataset_dir"])
        raise

    class_names: list[str] = original_ds.class_names

    try:
        model = load_model(ckpt_path)
    except (OSError, FileNotFoundError) as exc:
        logger.error("Model file not found: %s", ckpt_path)
        raise FileNotFoundError(f"Model file not found: {ckpt_path}") from exc

    logger.info("Evaluating model '%s' on validation set.", ckpt_path)

    y_true: list[int] = []
    y_pred: list[int] = []
    for X, y in val_ds:
        y_true.extend(np.argmax(y, 1))
        y_pred.extend(np.argmax(model.predict(X), 1))

    logger.info("\n%s", skm.classification_report(y_true, y_pred, target_names=class_names))

    cm = skm.confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix\n%s", cm)

    save_path = os.path.join(cfg["log_dir"], "evaluation", "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate a trained EfficientNet-B0 model.")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model_ckpt"],
        help="Path to the saved Keras model (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.model)
