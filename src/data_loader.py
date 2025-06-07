import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import CONFIG

def get_datasets(cfg=CONFIG):
    img_size = (cfg["img_height"], cfg["img_width"])
    common = dict(
        validation_split = cfg["split"]["val"],
        seed             = cfg["seed"],
        label_mode       = "categorical",
        image_size       = img_size,
        batch_size       = cfg["batch_size"],
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg["dataset_dir"], subset="training", **common)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        cfg["dataset_dir"], subset="validation", **common)

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def prep(ds, training=False):
        ds = ds.map(lambda x, y: (preprocess_input(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (aug(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    return prep(train_ds, True), prep(val_ds, False)
