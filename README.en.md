# Brain MRI Tumor Classification

**[한국어 버전](./README.md)**

I built this learning project to classify brain MRI images into four categories: **Glioma, Meningioma, Pituitary tumor, and Healthy**. It fine-tunes EfficientNet-B0 in two stages.

## Key Features

- **EfficientNet-B0 Backbone**: Starts from ImageNet pre-trained weights.
- **2-Stage Training Strategy**:
  - **Stage 1**: Frozen backbone to stabilize the custom classification head.
  - **Stage 2**: Fine-tuning selected layers to adapt the model to specific MRI features.
- **Mixed Precision Training**: Reduces GPU memory use and training time.
- **Training and Evaluation Scripts**: Connect data augmentation, training, and confusion-matrix generation.
- **Experiment Tracking**: TensorBoard records loss and accuracy during training.

## Tech Stack

- **Framework**: TensorFlow 2.10+, Keras
- **Model**: EfficientNet-B0 (Transfer Learning)
- **Language**: Python 3.8+
- **Metrics**: Scikit-learn (Precision, Recall, F1-Score)
- **Visualization**: Matplotlib, Seaborn, TensorBoard

## Project Structure

```text
src/
├── config.py       # Centralized hyperparameter management
├── data_loader.py  # Preprocessing & Data augmentation pipeline
├── model.py        # EfficientNet-B0 based architecture
├── train.py        # 2-stage training logic
└── evaluate.py     # Metrics & Confusion Matrix generation
```

## Technical Highlights

### 1. Two-stage fine-tuning
Stage 1 freezes the backbone and trains the classification head. Stage 2 unfreezes the top 20 layers for fine-tuning on the MRI data.

### 2. Preprocessing
The pipeline includes:
- Standardized resizing to 224x224.
- Dynamic data augmentation (Flip, Rotation, Zoom, Contrast) to handle limited dataset sizes and improve generalization.

## Getting Started

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

Hyperparameters, augmentation settings, and the model structure are documented in the [detailed manual](./DETAILS.en.md). This is a research and learning project, not a medical diagnostic tool.
