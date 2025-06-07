import tensorflow as tf
from tensorflow.keras import layers as L

def build_model(num_classes=4, img_height=224, img_width=224):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(img_height, img_width, 3))
    base.trainable = False

    inputs  = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base(inputs, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base
