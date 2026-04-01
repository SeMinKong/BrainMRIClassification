"""Live demo: display 10 sample Brain MRI images with (optional) model predictions."""

import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from src.config import CONFIG

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "healthy", "meningioma", "pituitary"]
N_IMAGES = 10


def collect_image_paths(dataset_dir: str) -> list[tuple[str, str]]:
    """Return list of (image_path, class_name) for all images in dataset_dir."""
    samples: list[tuple[str, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), class_name))
    return samples


def load_and_preprocess(path: str, img_height: int, img_width: int) -> np.ndarray:
    """Load an image, resize, and apply EfficientNet preprocessing."""
    img = Image.open(path).convert("RGB").resize((img_width, img_height))
    arr = np.array(img, dtype=np.float32)
    return preprocess_input(arr)


def main(ckpt_path: str | None, cfg: dict = CONFIG) -> None:
    dataset_dir: str = cfg["dataset_dir"]
    img_height: int = cfg["img_height"]
    img_width: int = cfg["img_width"]

    all_samples = collect_image_paths(dataset_dir)
    if not all_samples:
        raise FileNotFoundError(f"No images found in: {dataset_dir}")

    # Sample with replacement when fewer than N_IMAGES images are available
    chosen = random.choices(all_samples, k=N_IMAGES)

    model = None
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info("Loading model from: %s", ckpt_path)
        model = load_model(ckpt_path)
    elif ckpt_path:
        logger.warning("Model not found at '%s'. Showing true labels only.", ckpt_path)

    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    fig.suptitle("Brain MRI — Live Demo (10 samples)", fontsize=22, fontweight="bold")

    for ax, (img_path, true_label) in zip(axes.flat, chosen):
        # Display original (un-preprocessed) image
        raw = Image.open(img_path).convert("RGB").resize((img_width, img_height))
        ax.imshow(raw)
        ax.axis("off")

        if model is not None:
            arr = load_and_preprocess(img_path, img_height, img_width)
            probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
            pred_label = CLASS_NAMES[int(np.argmax(probs))]
            confidence = float(np.max(probs))
            color = "darkgreen" if pred_label == true_label else "darkred"
            title = f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})"
        else:
            color = "darkblue"
            title = f"True: {true_label}"

        ax.set_title(title, fontsize=13, fontweight="bold", color=color, pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(cfg["log_dir"], "evaluation", "demo_10_images.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logger.info("Demo saved to: %s", save_path)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Live demo: show 10 Brain MRI predictions.")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model_ckpt"],
        help="Path to saved Keras model (omit to show true labels only)",
    )
    args = parser.parse_args()
    main(args.model)
