# BrainMRIClassification - 뇌 MRI 종양 분류 시스템

> **한국어 문서 (Korean)**

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [사용자 가이드](#사용자-가이드)
- [개발자 가이드](#개발자-가이드)
- [English Documentation](#english-documentation)

---

## 프로젝트 개요

**BrainMRIClassification**은 EfficientNet-B0 백본에 전이학습(Transfer Learning)을 적용하여 뇌 MRI 이미지를 4개의 진단 클래스로 자동 분류하는 딥러닝 파이프라인입니다. 대학생 캡스톤, 의료 AI 연구, 머신러닝 실험용으로 설계되었으며, 간단한 설정 파일 수정만으로 즉시 실험을 시작할 수 있습니다.

**분류 대상 뇌 MRI 종양 클래스:**
1. **glioma** - 신경 조직에서 발생하는 종양
2. **pituitary** - 뇌하수체에서 발생하는 종양
3. **meningioma** - 뇌막에서 발생하는 종양
4. **healthy** - 정상 뇌 MRI (비종양)

---

## 주요 기능

- ✅ **사전 학습된 EfficientNet-B0** 모델 기반 전이학습
- ✅ **2단계 학습 전략** (Stage-1: 기본층 고정, Stage-2: 미세 조정)
- ✅ **혼합 정밀도 학습(Mixed Precision)** - GPU 메모리 절약 및 빠른 학습
- ✅ **데이터 증강** - RandomFlip, RandomRotation, RandomZoom, RandomContrast
- ✅ **자동 검증 분할** - train/validation 자동 분할 (70%/30%)
- ✅ **조기 종료 및 학습률 감소** - 과적합 방지
- ✅ **상세한 평가 지표** - Precision, Recall, F1-Score, 혼동 행렬(Confusion Matrix)
- ✅ **TensorBoard 로깅** - 실시간 학습 모니터링

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **프레임워크** | TensorFlow/Keras |
| **모델** | EfficientNet-B0 (ImageNet 사전 학습) |
| **언어** | Python 3.8+ |
| **이미지 처리** | TensorFlow Image Preprocessing |
| **평가** | scikit-learn (metrics, confusion_matrix) |
| **시각화** | Matplotlib, Seaborn |
| **로깅** | Python logging, TensorBoard |

---

## 사용자 가이드

### 전제 조건

- **Python 3.8 이상**
- **TensorFlow 2.10+** (GPU 버전 권장)
- **pip** 패키지 관리자

### 설치

#### 1. 저장소 클론
```bash
git clone <repository-url>
cd BrainMRIClassification
```

#### 2. 가상 환경 생성 (권장)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

#### 3. 필수 패키지 설치

```bash
pip install tensorflow>=2.10
pip install keras
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

또는 requirements.txt가 있는 경우:
```bash
pip install -r requirements.txt
```

### 데이터셋 준비

#### 데이터셋 디렉토리 구조

```
BrainMRI/
└── multi_class_dataset/
    ├── glioma/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    ├── pituitary/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    ├── meningioma/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    └── healthy/
        ├── img_1.jpg
        ├── img_2.jpg
        └── ...
```

**각 클래스 폴더에는 해당 클래스의 MRI 이미지 파일(.jpg, .png 등)이 포함되어야 합니다.**

#### 데이터셋 다운로드

데이터셋은 Kaggle의 [Brain MRI Images Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri-dataset)에서 다운로드할 수 있습니다.

### 설정 파일 수정

`src/config.py`의 `CONFIG` 딕셔너리를 프로젝트에 맞게 수정합니다:

```python
CONFIG: dict[str, Any] = {
    "dataset_dir": "BrainMRI/multi_class_dataset",  # 데이터셋 경로
    "img_height": 224,  # 입력 이미지 높이
    "img_width": 224,   # 입력 이미지 너비
    "batch_size": 32,   # 배치 크기
    "seed": 123,        # 난수 시드 (재현성)
    "split": {"train": 0.7, "val": 0.3},  # train/val 분할 비율
    "stage1": {"epochs": 30, "lr": 1e-3},  # Stage-1: 기본층 고정
    "stage2": {"epochs": 30, "lr": 1e-4, "unfreeze_layers": 20},  # Stage-2: 미세조정
    "mixed_precision": True,  # 혼합 정밀도 활성화
    "log_dir": "runs/brainMRI_effb0",  # 로그 및 체크포인트 저장 경로
    "model_ckpt": "best_effb0.keras",  # 최종 모델 저장 이름
}
```

### 모델 학습

#### 기본 학습 실행

```bash
python src/train.py
```

**학습 프로세스:**
1. 데이터셋 로드 및 전처리 (70% train, 30% validation)
2. **Stage-1**: EfficientNet-B0 기본층 고정 후 30에포크 학습
3. **Stage-2**: 마지막 20개 레이어 언프리즈 후 30에포크 미세 조정
4. 최고 성능 모델 자동 저장 (`best_effb0.keras`)

**학습 중 콘솔 출력 예시:**
```
2024-04-01 10:30:45 INFO train: Stage-1 training started.
Epoch 1/30
45/45 [==============================] - 120s 2s/step - loss: 1.8234 - accuracy: 0.4521 - val_loss: 1.5678 - val_accuracy: 0.5231
...
2024-04-01 11:45:23 INFO train: Stage-2 fine-tuning started.
Epoch 1/30
45/45 [==============================] - 130s 2s/step - loss: 0.8234 - accuracy: 0.7543 - val_loss: 0.6234 - val_accuracy: 0.8012
...
2024-04-01 13:20:56 INFO train: Training complete. Best model saved to 'best_effb0.keras'.
```

#### 커스텀 설정으로 학습

`src/train.py`를 수정하여 `CONFIG` 기본값을 오버라이드할 수 있습니다:

```python
from src.train import main
from src.config import CONFIG

custom_config = {
    **CONFIG,
    "batch_size": 16,
    "stage1": {"epochs": 20, "lr": 5e-4},
}
main(custom_config)
```

### 모델 평가

#### 기본 평가 실행

```bash
python src/evaluate.py --model best_effb0.keras
```

#### 커스텀 모델 경로 지정

```bash
python src/evaluate.py --model runs/custom_model.keras
```

**평가 출력:**
```
2024-04-01 13:25:30 INFO evaluate: Evaluating model 'best_effb0.keras' on validation set.

              precision    recall  f1-score   support

       glioma       0.89      0.87      0.88       235
     pituitary       0.91      0.92      0.91       210
    meningioma       0.85      0.86      0.86       198
       healthy       0.93      0.94      0.93       257
    
    accuracy                           0.90       900
   macro avg       0.90      0.90      0.90       900
weighted avg       0.90      0.90      0.90       900

2024-04-01 13:25:45 INFO evaluate: Confusion Matrix
[[205  12   8  10]
 [  8 193   5   4]
 [  7   6 170  15]
 [  4   3  12 238]]

2024-04-01 13:26:02 INFO evaluate: Confusion matrix saved to: runs/brainMRI_effb0/evaluation/confusion_matrix.png
```

### 결과 해석

#### 성능 지표

| 지표 | 의미 |
|------|------|
| **Accuracy** | 전체 정확도 (모든 예측 중 올바른 예측 비율) |
| **Precision** | 양성 예측의 정확도 (예측한 양성 중 실제 양성 비율) |
| **Recall** | 재현율 (실제 양성 중 올바르게 예측한 비율) |
| **F1-Score** | Precision과 Recall의 조화 평균 |

#### 혼동 행렬(Confusion Matrix)

- **대각선 원소**: 올바르게 분류된 샘플 수
- **비대각선 원소**: 오분류된 샘플 수

이미지는 `runs/brainMRI_effb0/evaluation/confusion_matrix.png`에 저장됩니다.

### 출력 파일

학습 및 평가 후 생성되는 파일들:

```
BrainMRIClassification/
├── best_effb0.keras  # 최고 성능 모델
├── runs/
│   └── brainMRI_effb0/
│       ├── events.out.tfevents.*  # TensorBoard 로그
│       └── evaluation/
│           └── confusion_matrix.png  # 혼동 행렬 시각화
```

#### TensorBoard 실시간 모니터링

```bash
tensorboard --logdir=runs/brainMRI_effb0
```

그 후 브라우저에서 `http://localhost:6006` 접속

---

## 개발자 가이드

### 프로젝트 구조

```
BrainMRIClassification/
├── src/
│   ├── __init__.py           # 패키지 초기화
│   ├── config.py             # 전역 설정 파일
│   ├── model.py              # 모델 정의
│   ├── data_loader.py        # 데이터 로딩 및 전처리
│   ├── train.py              # 학습 스크립트
│   └── evaluate.py           # 평가 스크립트
├── BrainMRI/
│   └── multi_class_dataset/  # 데이터셋 디렉토리
│       ├── glioma/
│       ├── pituitary/
│       ├── meningioma/
│       └── healthy/
├── runs/                     # 모델 체크포인트 및 로그
├── README.md                 # 이 문서
└── requirements.txt          # 의존성 패키지 목록
```

### 핵심 모듈 설명

#### 1. `src/config.py` - 설정 파일

모든 하이퍼파라미터를 한곳에서 관리합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `dataset_dir` | str | `BrainMRI/multi_class_dataset` | 데이터셋 루트 경로 |
| `img_height` | int | `224` | 입력 이미지 높이 (픽셀) |
| `img_width` | int | `224` | 입력 이미지 너비 (픽셀) |
| `batch_size` | int | `32` | 배치 크기 |
| `seed` | int | `123` | 난수 시드 (재현성) |
| `split.train` | float | `0.7` | 학습 데이터 비율 |
| `split.val` | float | `0.3` | 검증 데이터 비율 |
| `stage1.epochs` | int | `30` | Stage-1 에포크 수 |
| `stage1.lr` | float | `1e-3` | Stage-1 학습률 |
| `stage2.epochs` | int | `30` | Stage-2 에포크 수 |
| `stage2.lr` | float | `1e-4` | Stage-2 학습률 (보통 더 작음) |
| `stage2.unfreeze_layers` | int | `20` | Stage-2에서 언프리즈할 레이어 수 |
| `mixed_precision` | bool | `True` | 혼합 정밀도(float16) 활성화 |
| `log_dir` | str | `runs/brainMRI_effb0` | 로그 및 체크포인트 저장 경로 |
| `model_ckpt` | str | `best_effb0.keras` | 저장할 모델 파일명 |

#### 2. `src/model.py` - 모델 아키텍처

EfficientNet-B0 기반 분류 모델을 정의합니다.

**함수 시그니처:**
```python
def build_model(
    num_classes: int = 4,
    img_height: int = 224,
    img_width: int = 224,
) -> tuple[tf.keras.Model, tf.keras.Model]:
```

**모델 구조:**
```
Input (224x224x3)
    ↓
EfficientNet-B0 (ImageNet 사전학습, 초기에는 고정)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(num_classes=4, activation='softmax')
    ↓
Output (4개 클래스 확률)
```

**반환값:**
- `model`: 전체 모델 (평가/예측용)
- `base`: 백본 모델 (Stage-2에서 레이어 언프리즈용)

#### 3. `src/data_loader.py` - 데이터 로딩

데이터셋 로드, 전처리, 증강을 담당합니다.

**핵심 함수:**

```python
def get_datasets(cfg: dict = CONFIG) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """학습/검증 데이터셋 로드"""
    # 반환: (train_dataset, validation_dataset)
```

**전처리 파이프라인:**
1. EfficientNet용 표준 정규화 (`preprocess_input`)
2. **학습 데이터 증강** (training=True인 경우):
   - RandomFlip (수평 뒤집기)
   - RandomRotation (0~10도)
   - RandomZoom (90~110% 확대/축소)
   - RandomContrast (대비 변화)
3. 배치 처리 및 프리페칭 (성능 최적화)

#### 4. `src/train.py` - 학습 스크립트

2단계 학습 전략을 구현합니다.

**실행 방식:**
```bash
python src/train.py
```

**Stage-1 (기본층 고정):**
- EfficientNet-B0의 모든 레이어 고정
- 분류 헤드(Dense layer) 학습
- 높은 학습률 사용 (1e-3)

**Stage-2 (미세 조정):**
- 마지막 20개 레이어 언프리즈
- 전체 모델 재학습
- 낮은 학습률 사용 (1e-4)
- 콜백 사용:
  - `EarlyStopping`: 3 에포크 개선 없으면 중지
  - `ReduceLROnPlateau`: 2 에포크 개선 없으면 학습률 감소
  - `ModelCheckpoint`: 최고 성능 모델 저장
  - `TensorBoard`: 실시간 로깅

#### 5. `src/evaluate.py` - 평가 스크립트

학습된 모델을 평가합니다.

**실행 방식:**
```bash
python src/evaluate.py --model best_effb0.keras
```

**평가 항목:**
- Classification Report (precision, recall, f1-score)
- Confusion Matrix (텍스트 및 이미지)

---

### 모델 아키텍처 상세

#### EfficientNet-B0 개요

- **사전 학습**: ImageNet (1,000개 클래스)
- **파라미터**: ~4.2M
- **입력 크기**: 224x224x3
- **특징**: 높은 정확도와 효율성의 균형

#### 커스텀 분류 헤드

```
EfficientNet-B0 (출력: 7x7x1280)
    ↓
GlobalAveragePooling2D (출력: 1280)
    ↓
Dropout(0.3) (정규화)
    ↓
Dense(4, softmax) (분류: 4개 클래스)
```

**각 레이어의 역할:**
- **GlobalAveragePooling2D**: 공간 차원 제거, 특징 벡터 추출
- **Dropout**: 과적합 방지
- **Dense**: 4개 클래스로의 확률 변환

---

### 하이퍼파라미터 튜닝 가이드

#### 학습률(Learning Rate) 조정

- Stage-1: `1e-3` (기본값) - 분류 헤드가 처음이므로 높은 학습률
- Stage-2: `1e-4` (기본값) - 사전 학습된 가중치를 보존하므로 낮은 학습률

**조정 방법:**
```python
CONFIG["stage1"]["lr"] = 5e-4  # 더 낮게
CONFIG["stage2"]["lr"] = 5e-5  # 더 낮게
```

#### 배치 크기(Batch Size) 조정

- **기본값**: 32
- **감소**: GPU 메모리 부족 시 (16, 8)
- **증가**: 큰 GPU 메모리 있을 때 (64, 128)

#### 에포크(Epochs) 조정

- **기본값**: Stage-1 30, Stage-2 30
- **감소**: 빠른 테스트 (10, 10)
- **증가**: 더 정확한 학습 (50, 50)

#### Early Stopping 조정 (`src/train.py`)

```python
EarlyStopping(patience=3, restore_best_weights=True)
```

- `patience=3`: 3 에포크 동안 개선 없으면 중지
- 값을 늘리면 더 오래 학습, 줄이면 빨리 중지

---

### 새로운 클래스 추가

현재는 4개 클래스(glioma, pituitary, meningioma, healthy)로 설정되어 있습니다.

**클래스 추가 방법:**

1. **데이터셋 구조 수정**:
   ```
   BrainMRI/multi_class_dataset/
   ├── glioma/
   ├── pituitary/
   ├── meningioma/
   ├── healthy/
   └── new_class/  # 새 클래스 폴더 추가
   ```

2. **src/model.py 수정** (필요시):
   ```python
   model, base = build_model(
       num_classes=5,  # 4에서 5로 변경
       img_height=cfg["img_height"],
       img_width=cfg["img_width"],
   )
   ```

3. **학습 실행**:
   ```bash
   python src/train.py
   ```

데이터 로더가 자동으로 클래스 폴더를 인식하고 원-핫 인코딩합니다.

---

### 일반적인 문제 해결

#### 문제: `FileNotFoundError: Dataset directory not found`
**해결:**
```bash
# 데이터셋 경로 확인
ls -la BrainMRI/multi_class_dataset/
# config.py의 dataset_dir이 올바른지 확인
```

#### 문제: GPU 메모리 부족 (OOM)
**해결:**
```python
CONFIG["batch_size"] = 16  # 32에서 16으로 감소
CONFIG["mixed_precision"] = True  # 이미 활성화
```

#### 문제: 정확도가 낮음
**시도:**
```python
# 1. 학습 에포크 증가
CONFIG["stage1"]["epochs"] = 50
CONFIG["stage2"]["epochs"] = 50

# 2. 데이터 증강 강도 높이기 (data_loader.py)
tf.keras.layers.RandomRotation(0.2)  # 0.1에서 0.2로

# 3. 더 많은 레이어 언프리즈
CONFIG["stage2"]["unfreeze_layers"] = 50  # 20에서 50으로
```

---

## English Documentation

> **English Version**

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [User Guide](#user-guide)
- [Developer Guide](#developer-guide)

---

## Project Overview

**BrainMRIClassification** is a deep learning pipeline that applies transfer learning using the EfficientNet-B0 backbone to automatically classify brain MRI images into 4 diagnostic classes. Designed for university capstone projects, medical AI research, and machine learning experiments, it requires only simple configuration file modifications to start experiments immediately.

**Brain MRI Tumor Classification Classes:**
1. **glioma** - Tumors originating from neural tissue
2. **pituitary** - Tumors originating from the pituitary gland
3. **meningioma** - Tumors originating from brain membranes
4. **healthy** - Normal brain MRI (non-tumor)

---

## Features

- ✅ **Pre-trained EfficientNet-B0** backbone for transfer learning
- ✅ **2-Stage Training Strategy** (Stage-1: frozen base, Stage-2: fine-tuning)
- ✅ **Mixed Precision Training** - GPU memory savings and faster learning
- ✅ **Data Augmentation** - RandomFlip, RandomRotation, RandomZoom, RandomContrast
- ✅ **Automatic Validation Split** - Train/validation automatic split (70%/30%)
- ✅ **Early Stopping & Learning Rate Reduction** - Overfitting prevention
- ✅ **Detailed Evaluation Metrics** - Precision, Recall, F1-Score, Confusion Matrix
- ✅ **TensorBoard Logging** - Real-time training monitoring

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | TensorFlow/Keras |
| **Model** | EfficientNet-B0 (ImageNet pre-trained) |
| **Language** | Python 3.8+ |
| **Image Processing** | TensorFlow Image Preprocessing |
| **Evaluation** | scikit-learn (metrics, confusion_matrix) |
| **Visualization** | Matplotlib, Seaborn |
| **Logging** | Python logging, TensorBoard |

---

## User Guide

### Prerequisites

- **Python 3.8 or higher**
- **TensorFlow 2.10+** (GPU version recommended)
- **pip** package manager

### Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd BrainMRIClassification
```

#### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

#### 3. Install Required Packages

```bash
pip install tensorflow>=2.10
pip install keras
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

Or if requirements.txt exists:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

#### Dataset Directory Structure

```
BrainMRI/
└── multi_class_dataset/
    ├── glioma/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    ├── pituitary/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    ├── meningioma/
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ...
    └── healthy/
        ├── img_1.jpg
        ├── img_2.jpg
        └── ...
```

**Each class folder must contain MRI image files (.jpg, .png, etc.) of that class.**

#### Download Dataset

The dataset can be downloaded from Kaggle's [Brain MRI Images Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri-dataset).

### Configuration File

Modify the `CONFIG` dictionary in `src/config.py` to match your project:

```python
CONFIG: dict[str, Any] = {
    "dataset_dir": "BrainMRI/multi_class_dataset",  # Dataset path
    "img_height": 224,  # Input image height
    "img_width": 224,   # Input image width
    "batch_size": 32,   # Batch size
    "seed": 123,        # Random seed (reproducibility)
    "split": {"train": 0.7, "val": 0.3},  # Train/val split ratio
    "stage1": {"epochs": 30, "lr": 1e-3},  # Stage-1: base layers frozen
    "stage2": {"epochs": 30, "lr": 1e-4, "unfreeze_layers": 20},  # Stage-2: fine-tuning
    "mixed_precision": True,  # Enable mixed precision
    "log_dir": "runs/brainMRI_effb0",  # Logs and checkpoint directory
    "model_ckpt": "best_effb0.keras",  # Final model filename
}
```

### Model Training

#### Basic Training

```bash
python src/train.py
```

**Training Process:**
1. Load and preprocess dataset (70% train, 30% validation)
2. **Stage-1**: Train with frozen EfficientNet-B0 base layers for 30 epochs
3. **Stage-2**: Unfreeze last 20 layers and fine-tune for 30 epochs
4. Automatically save best-performing model (`best_effb0.keras`)

**Example Console Output:**
```
2024-04-01 10:30:45 INFO train: Stage-1 training started.
Epoch 1/30
45/45 [==============================] - 120s 2s/step - loss: 1.8234 - accuracy: 0.4521 - val_loss: 1.5678 - val_accuracy: 0.5231
...
2024-04-01 11:45:23 INFO train: Stage-2 fine-tuning started.
Epoch 1/30
45/45 [==============================] - 130s 2s/step - loss: 0.8234 - accuracy: 0.7543 - val_loss: 0.6234 - val_accuracy: 0.8012
...
2024-04-01 13:20:56 INFO train: Training complete. Best model saved to 'best_effb0.keras'.
```

#### Custom Configuration Training

Modify `src/train.py` to override default `CONFIG`:

```python
from src.train import main
from src.config import CONFIG

custom_config = {
    **CONFIG,
    "batch_size": 16,
    "stage1": {"epochs": 20, "lr": 5e-4},
}
main(custom_config)
```

### Model Evaluation

#### Basic Evaluation

```bash
python src/evaluate.py --model best_effb0.keras
```

#### Custom Model Path

```bash
python src/evaluate.py --model runs/custom_model.keras
```

**Evaluation Output:**
```
2024-04-01 13:25:30 INFO evaluate: Evaluating model 'best_effb0.keras' on validation set.

              precision    recall  f1-score   support

       glioma       0.89      0.87      0.88       235
     pituitary       0.91      0.92      0.91       210
    meningioma       0.85      0.86      0.86       198
       healthy       0.93      0.94      0.93       257
    
    accuracy                           0.90       900
   macro avg       0.90      0.90      0.90       900
weighted avg       0.90      0.90      0.90       900

2024-04-01 13:25:45 INFO evaluate: Confusion Matrix
[[205  12   8  10]
 [  8 193   5   4]
 [  7   6 170  15]
 [  4   3  12 238]]

2024-04-01 13:26:02 INFO evaluate: Confusion matrix saved to: runs/brainMRI_effb0/evaluation/confusion_matrix.png
```

### Interpreting Results

#### Performance Metrics

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Overall correctness (correct predictions / total predictions) |
| **Precision** | Accuracy of positive predictions (true positives / predicted positives) |
| **Recall** | Coverage of actual positives (true positives / actual positives) |
| **F1-Score** | Harmonic mean of Precision and Recall |

#### Confusion Matrix

- **Diagonal elements**: Correctly classified samples
- **Off-diagonal elements**: Misclassified samples

Image saved to `runs/brainMRI_effb0/evaluation/confusion_matrix.png`.

### Output Files

Files generated after training and evaluation:

```
BrainMRIClassification/
├── best_effb0.keras  # Best performing model
├── runs/
│   └── brainMRI_effb0/
│       ├── events.out.tfevents.*  # TensorBoard logs
│       └── evaluation/
│           └── confusion_matrix.png  # Confusion matrix visualization
```

#### Real-time Monitoring with TensorBoard

```bash
tensorboard --logdir=runs/brainMRI_effb0
```

Then open `http://localhost:6006` in your browser.

---

## Developer Guide

### Project Structure

```
BrainMRIClassification/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Global configuration
│   ├── model.py              # Model definition
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── BrainMRI/
│   └── multi_class_dataset/  # Dataset directory
│       ├── glioma/
│       ├── pituitary/
│       ├── meningioma/
│       └── healthy/
├── runs/                     # Model checkpoints and logs
├── README.md                 # This document
└── requirements.txt          # Dependency packages
```

### Core Modules

#### 1. `src/config.py` - Configuration File

Manages all hyperparameters in a single location.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_dir` | str | `BrainMRI/multi_class_dataset` | Dataset root path |
| `img_height` | int | `224` | Input image height (pixels) |
| `img_width` | int | `224` | Input image width (pixels) |
| `batch_size` | int | `32` | Batch size |
| `seed` | int | `123` | Random seed (reproducibility) |
| `split.train` | float | `0.7` | Training data ratio |
| `split.val` | float | `0.3` | Validation data ratio |
| `stage1.epochs` | int | `30` | Stage-1 epochs |
| `stage1.lr` | float | `1e-3` | Stage-1 learning rate |
| `stage2.epochs` | int | `30` | Stage-2 epochs |
| `stage2.lr` | float | `1e-4` | Stage-2 learning rate (usually lower) |
| `stage2.unfreeze_layers` | int | `20` | Number of layers to unfreeze in Stage-2 |
| `mixed_precision` | bool | `True` | Enable mixed precision (float16) |
| `log_dir` | str | `runs/brainMRI_effb0` | Logs and checkpoint directory |
| `model_ckpt` | str | `best_effb0.keras` | Model filename to save |

#### 2. `src/model.py` - Model Architecture

Defines the EfficientNet-B0 based classification model.

**Function Signature:**
```python
def build_model(
    num_classes: int = 4,
    img_height: int = 224,
    img_width: int = 224,
) -> tuple[tf.keras.Model, tf.keras.Model]:
```

**Model Structure:**
```
Input (224x224x3)
    ↓
EfficientNet-B0 (ImageNet pre-trained, frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(num_classes=4, activation='softmax')
    ↓
Output (4 class probabilities)
```

**Return Values:**
- `model`: Full model (for evaluation/prediction)
- `base`: Backbone model (for unfreezing layers in Stage-2)

#### 3. `src/data_loader.py` - Data Loading

Handles dataset loading, preprocessing, and augmentation.

**Core Function:**

```python
def get_datasets(cfg: dict = CONFIG) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train and validation datasets"""
    # Returns: (train_dataset, validation_dataset)
```

**Preprocessing Pipeline:**
1. EfficientNet standard normalization (`preprocess_input`)
2. **Training Data Augmentation** (when training=True):
   - RandomFlip (horizontal flip)
   - RandomRotation (0~10 degrees)
   - RandomZoom (90~110% zoom)
   - RandomContrast (contrast variation)
3. Batching and prefetching (performance optimization)

#### 4. `src/train.py` - Training Script

Implements the 2-stage training strategy.

**Execution:**
```bash
python src/train.py
```

**Stage-1 (Frozen Base):**
- Freeze all EfficientNet-B0 layers
- Train classification head (Dense layer) only
- Use high learning rate (1e-3)

**Stage-2 (Fine-tuning):**
- Unfreeze last 20 layers
- Retrain entire model
- Use lower learning rate (1e-4)
- Callbacks:
  - `EarlyStopping`: Stop if no improvement for 3 epochs
  - `ReduceLROnPlateau`: Reduce LR if no improvement for 2 epochs
  - `ModelCheckpoint`: Save best model
  - `TensorBoard`: Real-time logging

#### 5. `src/evaluate.py` - Evaluation Script

Evaluates a trained model.

**Execution:**
```bash
python src/evaluate.py --model best_effb0.keras
```

**Evaluation Items:**
- Classification Report (precision, recall, f1-score)
- Confusion Matrix (text and image)

---

### Model Architecture Details

#### EfficientNet-B0 Overview

- **Pre-training**: ImageNet (1,000 classes)
- **Parameters**: ~4.2M
- **Input Size**: 224x224x3
- **Advantage**: Balance of high accuracy and efficiency

#### Custom Classification Head

```
EfficientNet-B0 (output: 7x7x1280)
    ↓
GlobalAveragePooling2D (output: 1280)
    ↓
Dropout(0.3) (regularization)
    ↓
Dense(4, softmax) (classification: 4 classes)
```

**Layer Roles:**
- **GlobalAveragePooling2D**: Remove spatial dimensions, extract feature vector
- **Dropout**: Prevent overfitting
- **Dense**: Convert to 4 class probabilities

---

### Hyperparameter Tuning Guide

#### Learning Rate Adjustment

- Stage-1: `1e-3` (default) - High LR for newly trained classification head
- Stage-2: `1e-4` (default) - Low LR to preserve pre-trained weights

**Adjustment Example:**
```python
CONFIG["stage1"]["lr"] = 5e-4  # Lower
CONFIG["stage2"]["lr"] = 5e-5  # Lower
```

#### Batch Size Adjustment

- **Default**: 32
- **Decrease**: Out of GPU memory (16, 8)
- **Increase**: Large GPU memory available (64, 128)

#### Epochs Adjustment

- **Default**: Stage-1 30, Stage-2 30
- **Decrease**: Quick testing (10, 10)
- **Increase**: More accurate training (50, 50)

#### Early Stopping Adjustment (`src/train.py`)

```python
EarlyStopping(patience=3, restore_best_weights=True)
```

- `patience=3`: Stop if no improvement for 3 epochs
- Increase to train longer, decrease to stop faster

---

### Adding New Classes

Currently configured for 4 classes (glioma, pituitary, meningioma, healthy).

**Steps to Add Classes:**

1. **Modify Dataset Structure**:
   ```
   BrainMRI/multi_class_dataset/
   ├── glioma/
   ├── pituitary/
   ├── meningioma/
   ├── healthy/
   └── new_class/  # Add new class folder
   ```

2. **Modify `src/model.py` (if needed)**:
   ```python
   model, base = build_model(
       num_classes=5,  # Change from 4 to 5
       img_height=cfg["img_height"],
       img_width=cfg["img_width"],
   )
   ```

3. **Run Training**:
   ```bash
   python src/train.py
   ```

The data loader will automatically recognize class folders and apply one-hot encoding.

---

### Troubleshooting

#### Issue: `FileNotFoundError: Dataset directory not found`
**Solution:**
```bash
# Verify dataset path
ls -la BrainMRI/multi_class_dataset/
# Check dataset_dir in config.py
```

#### Issue: GPU Out of Memory (OOM)
**Solution:**
```python
CONFIG["batch_size"] = 16  # Reduce from 32 to 16
CONFIG["mixed_precision"] = True  # Already enabled
```

#### Issue: Low Accuracy
**Try:**
```python
# 1. Increase training epochs
CONFIG["stage1"]["epochs"] = 50
CONFIG["stage2"]["epochs"] = 50

# 2. Strengthen data augmentation (data_loader.py)
tf.keras.layers.RandomRotation(0.2)  # Increase from 0.1 to 0.2

# 3. Unfreeze more layers
CONFIG["stage2"]["unfreeze_layers"] = 50  # Increase from 20 to 50
```

---

**Last Updated**: 2024-04-01  
**Maintainer**: SSAFY BrainMRI Classification Project Team
