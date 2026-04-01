import logging
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from config import CONFIG
from data_loader import get_datasets
from model import build_model

logger = logging.getLogger(__name__)


def main(cfg: dict = CONFIG) -> None:
    if cfg.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, val_ds = get_datasets(cfg)
    model, base = build_model(
        img_height=cfg["img_height"],
        img_width=cfg["img_width"],
    )

    # ───────── Stage-1 ─────────
    logger.info("Stage-1 training started.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["stage1"]["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["stage1"]["epochs"],
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )

    # ───────── Stage-2 ─────────
    logger.info("Stage-2 fine-tuning started.")
    for layer in base.layers[-cfg["stage2"]["unfreeze_layers"]:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["stage2"]["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2),
        ModelCheckpoint(cfg["model_ckpt"], save_best_only=True),
        TensorBoard(log_dir=cfg["log_dir"]),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["stage2"]["epochs"],
        callbacks=cbs,
    )
    logger.info("Training complete. Best model saved to '%s'.", cfg["model_ckpt"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
