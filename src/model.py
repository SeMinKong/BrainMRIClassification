import logging

import tensorflow as tf
from tensorflow.keras import layers as L

logger = logging.getLogger(__name__)


def build_model(
    num_classes: int = 4,
    img_height: int = 224,
    img_width: int = 224,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build an EfficientNetB0-based classification model.

    Returns:
        A tuple of (full_model, base_model) where base_model is the
        EfficientNetB0 backbone (frozen by default for stage-1 training).
    """
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base(inputs, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)

    logger.info(
        "Model built: EfficientNetB0 backbone, %d output classes, input %dx%d.",
        num_classes,
        img_height,
        img_width,
    )
    return tf.keras.Model(inputs, outputs), base
