# Brain MRI 종양 분류 시스템

**[English Version](./README.en.md)**

뇌 MRI 영상을 분석하여 **신경교종(Glioma), 수막종(Meningioma), 뇌하수체 종양(Pituitary tumor), 정상(Healthy)** 4개 클래스로 분류하는 딥러닝 파이프라인입니다. EfficientNet-B0 모델을 활용한 전이학습(Transfer Learning)과 정교한 2단계 학습 전략을 통해 높은 진단 정확도를 확보하는 데 주력했습니다.

## 주요 특징

- **EfficientNet-B0 백본 활용**: ImageNet으로 사전 학습된 최신 아키텍처를 사용하여 이미지 특징 추출 성능을 극대화했습니다.
- **2단계 학습 전략(2-Stage Training)**:
    - **Stage 1**: 백본을 고정한 채 분류 헤드(Classification Head)를 먼저 안정화합니다.
    - **Stage 2**: 상위 20개 레이어를 언프리즈(Unfreeze)하여 MRI 영상 특성에 맞게 미세 조정(Fine-tuning)합니다.
- **혼합 정밀도 학습(Mixed Precision)**: GPU 메모리 효율을 높이고 학습 속도를 획기적으로 개선했습니다.
- **자동화된 파이프라인**: 데이터 증강부터 학습, 평가(Confusion Matrix 생성)까지의 전 과정을 자동화했습니다.
- **실시간 모니터링**: TensorBoard를 연동하여 손실값과 정확도의 변화를 실시간으로 추적할 수 있습니다.

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
의료 데이터의 특성을 반영하면서도 기존 모델의 강력한 특징 추출 능력을 유지하기 위해 노력했습니다. 특히 Stage 2에서 레이어 일부만을 선택적으로 학습시키는 전략을 통해 과적합(Overfitting)을 방지하고 안정적인 수렴을 이끌어냈습니다.

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

>  **더 자세한 정보가 필요하신가요?**
> 상세한 모델 아키텍처 설계, 하이퍼파라미터 파인튜닝 가이드 및 데이터 증강 전략은 [상세 매뉴얼(DETAILS.md)](./DETAILS.md)에서 확인하실 수 있습니다.

---
의료 AI 연구 및 학습 목적으로 개발되었습니다.
