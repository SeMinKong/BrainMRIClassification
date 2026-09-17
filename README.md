# Brain MRI 종양 분류 시스템

**[English Version](./README.en.md)**

**DEMO**
<img width="2700" height="1500" alt="demo_10_images" src="https://github.com/user-attachments/assets/6628d795-cd0d-424b-99ce-9ebbda64e1c9" />

뇌 MRI 영상을 **신경교종(Glioma), 수막종(Meningioma), 뇌하수체 종양(Pituitary tumor), 정상(Healthy)** 4개 클래스로 분류하는 학습 프로젝트입니다. EfficientNet-B0 전이학습을 두 단계로 나눠 구현했습니다.

## 주요 특징

- **EfficientNet-B0 백본**: ImageNet 사전 학습 가중치에서 시작합니다.
- **2단계 학습 전략(2-Stage Training)**:
    - **Stage 1**: 백본을 고정한 채 분류 헤드(Classification Head)를 먼저 안정화합니다.
    - **Stage 2**: 상위 20개 레이어를 언프리즈(Unfreeze)하여 MRI 영상 특성에 맞게 미세 조정(Fine-tuning)합니다.
- **혼합 정밀도 학습(Mixed Precision)**: GPU 메모리 사용량과 학습 시간을 줄이기 위해 적용했습니다.
- **학습·평가 흐름**: 데이터 증강, 학습, 혼동 행렬 생성을 스크립트로 연결했습니다.
- **학습 기록**: TensorBoard에서 손실값과 정확도 변화를 확인할 수 있습니다.

## 기술 스택

- **프레임워크**: TensorFlow 2.10+, Keras
- **모델**: EfficientNet-B0 (전이학습 적용)
- **언어**: Python 3.8+
- **이미지 처리**: TensorFlow Image Preprocessing
- **평가 및 시각화**: Scikit-learn, Matplotlib, Seaborn

## 프로젝트 구조

```text
src/
├── config.py       # 하이퍼파라미터 및 전역 설정 관리
├── data_loader.py  # 데이터 증강 및 전처리 파이프라인
├── model.py        # EfficientNet-B0 기반 모델 정의
├── train.py        # 2단계 학습 및 콜백 로직
└── evaluate.py     # 성능 지표 및 혼동 행렬 시각화
```

## 핵심 기술 구현 내용

### 1. 전이학습 및 미세 조정(Fine-tuning)
Stage 1에서는 백본을 고정하고 분류 헤드만 학습합니다. Stage 2에서는 상위 20개 레이어를 풀어 MRI 데이터에 맞게 미세 조정합니다.

### 2. 데이터 전처리
데이터셋의 크기가 제한적인 의료 AI 환경에서 모델의 일반화 성능을 높이기 위해 다음 기법을 적용했습니다:
- 표준 224x224 리사이징 및 정규화
- 다양한 증강 기법(반전, 회전, 줌, 대비 조정)을 통한 데이터 다양성 확보

## 시작하기

### 설치 방법
```bash
git clone <repository-url>
cd BrainMRIClassification
pip install -r requirements.txt
```

### 모델 학습
`src/config.py`에서 데이터셋 경로를 설정한 후 다음 명령어를 실행합니다:
```bash
python src/train.py
```

### 모델 평가
```bash
python src/evaluate.py --model best_effb0.keras
```

모델 구조, 하이퍼파라미터, 데이터 증강 설정은 [상세 매뉴얼](./DETAILS.md)에 정리했습니다. 이 프로젝트는 연구·학습용이며 의료 진단에 사용하지 않습니다.
