import numpy as np, argparse
import sklearn.metrics as skm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from src.config import CONFIG
from src.data_loader import get_datasets

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main(ckpt_path, cfg= CONFIG):
    # 원본 데이터셋 가져오기
    train_ds, val_ds = get_datasets(cfg)
    
    # 원본 데이터셋에서 클래스 이름 가져오기
    original_ds = tf.keras.utils.image_dataset_from_directory(
        cfg["dataset_dir"],
        validation_split=cfg["split"]["val"],
        subset="training",
        seed=cfg["seed"],
        image_size=(cfg["img_height"], cfg["img_width"]),
        batch_size=cfg["batch_size"]
    )
    class_names = original_ds.class_names
    
    model = load_model(ckpt_path)

    y_true, y_pred = [], []
    for X, y in val_ds:
        y_true.extend(np.argmax(y, 1))
        y_pred.extend(np.argmax(model.predict(X), 1))

    # 분류 보고서 출력
    print(skm.classification_report(
        y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix 계산
    cm = skm.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix\n", cm)
    
    # Confusion Matrix 시각화 및 저장
    save_path = os.path.join(cfg["log_dir"], "evaluation", "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path)
    print(f"Confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    main("best_effb0.keras")
