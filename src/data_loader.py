import logging
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import CONFIG

logger = logging.getLogger(__name__)


def prepare_dataset(
    ds: tf.data.Dataset,
    aug: tf.keras.Sequential,
    training: bool = False,
) -> tf.data.Dataset:
    """Apply preprocessing (and optional augmentation) to a dataset."""
    ds = ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def get_datasets(cfg: dict = CONFIG) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and preprocess train/validation datasets from the configured directory."""
    dataset_dir: str = cfg["dataset_dir"]
    img_size: tuple[int, int] = (cfg["img_height"], cfg["img_width"])
    common: dict = dict(
        validation_split=cfg["split"]["val"],
        seed=cfg["seed"],
        label_mode="categorical",
        image_size=img_size,
        batch_size=cfg["batch_size"],
    )

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir, subset="training", **common
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir, subset="validation", **common
        )
    except FileNotFoundError as exc:
        logger.error("Dataset directory not found: %s", dataset_dir)
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        ) from exc

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    logger.info("Datasets loaded from '%s'.", dataset_dir)
    return prepare_dataset(train_ds, aug, training=True), prepare_dataset(val_ds, aug, training=False)
