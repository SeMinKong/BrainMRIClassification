import tensorflow as tf, pathlib, argparse, os
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
from src.config import CONFIG
from src.data_loader import get_datasets
from src.model import build_model

def main(cfg=CONFIG):
    if cfg.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, val_ds = get_datasets(cfg)
    model, base = build_model(img_height=cfg["img_height"],
                              img_width=cfg["img_width"])
    # ───────── Stage-1 ─────────
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["stage1"]["lr"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg["stage1"]["epochs"],
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    # ───────── Stage-2 ─────────
    for layer in base.layers[-cfg["stage2"]["unfreeze_layers"]:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["stage2"]["lr"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2),
        ModelCheckpoint(cfg["model_ckpt"], save_best_only=True),
        TensorBoard(log_dir=cfg["log_dir"]),
    ]
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg["stage2"]["epochs"], callbacks=cbs)

if __name__ == "__main__":
    main()
