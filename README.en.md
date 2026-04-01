# Brain MRI Tumor Classification 🧠

**[한국어 버전](./README.md)**

A deep learning pipeline designed to classify brain MRI images into four categories: **Glioma, Meningioma, Pituitary tumor, and Healthy**. This project leverages transfer learning with EfficientNet-B0 and a specialized 2-stage training strategy to achieve high diagnostic accuracy.

## 🚀 Key Features

- **EfficientNet-B0 Backbone**: Utilizes a pre-trained state-of-the-art model for robust feature extraction.
- **2-Stage Training Strategy**:
  - **Stage 1**: Frozen backbone to stabilize the custom classification head.
  - **Stage 2**: Fine-tuning selected layers to adapt the model to specific MRI features.
- **Mixed Precision Training**: Optimized for GPU memory efficiency and faster convergence.
- **Automated Pipeline**: End-to-end support from data augmentation and preprocessing to evaluation with Confusion Matrices.
- **Experimental Tracking**: Integrated with TensorBoard for real-time monitoring of loss and accuracy.

## 🛠 Tech Stack

- **Framework**: TensorFlow 2.10+, Keras
- **Model**: EfficientNet-B0 (Transfer Learning)
- **Language**: Python 3.8+
- **Metrics**: Scikit-learn (Precision, Recall, F1-Score)
- **Visualization**: Matplotlib, Seaborn, TensorBoard

## 🏗 Project Structure

```text
src/
├── config.py       # Centralized hyperparameter management
├── data_loader.py  # Preprocessing & Data augmentation pipeline
├── model.py        # EfficientNet-B0 based architecture
├── train.py        # 2-stage training logic
└── evaluate.py     # Metrics & Confusion Matrix generation
```

## 🧠 Technical Highlights

### 1. Fine-tuning with "Unfreeze" Strategy
To preserve the powerful general features of EfficientNet while adapting to medical imaging nuances, I implemented a strategy to unfreeze only the top 20 layers during Stage 2. This prevents catastrophic forgetting and ensures stable convergence.

### 2. Robust Preprocessing
Medical images require careful handling. The pipeline includes:
- Standardized resizing to 224x224.
- Dynamic data augmentation (Flip, Rotation, Zoom, Contrast) to handle limited dataset sizes and improve generalization.

## 🏁 Getting Started

### Installation
```bash
git clone <repository-url>
cd BrainMRIClassification
pip install -r requirements.txt
```

### Training
Configure your dataset path in `src/config.py`, then run:
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py --model best_effb0.keras
```

> 💡 **Need more details?**
> For hyperparameters, dynamic data augmentation strategies, and architectural specifics, please refer to the [Detailed Manual (DETAILS.en.md)](./DETAILS.en.md).

---
Developed for Medical AI Research & Capstone Projects.
