# 모든 하이퍼파라미터를 dict 하나에 보관
CONFIG = {
    "dataset_dir": "BrainMRI/multi_class_dataset",
    "img_height": 224,
    "img_width": 224,
    "batch_size": 32,
    "seed": 123,
    "split": {"train": 0.7, "val": 0.3},
    "stage1": {"epochs": 30, "lr": 1e-3},
    "stage2": {"epochs": 30, "lr": 1e-4, "unfreeze_layers": 20},
    "mixed_precision": True,
    "log_dir": "runs/brainMRI_effb0",
    "model_ckpt": "best_effb0.keras",
}
