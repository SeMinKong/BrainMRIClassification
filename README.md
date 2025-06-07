# BrainMRIClassification

BrainMRI Classification은 EfficientNet-B0 백본에 전이학습(transfer learning)을 적용하여
뇌 MRI 이미지를 4개 진단 클래스로 자동 분류하는 딥러닝 파이프라인입니다.
대학생 캡스톤 및 연구 실험용으로 설계되어, 원-클릭 학습/평가 스크립트와
직관적인 config.py 설정 파일만 수정하면 바로 실험을 돌려볼 수 있습니다.

BrainMRIClassification/
├── BrainMRI/                     # 원본 데이터 루트
│   └── multi_class_dataset/      # class1/ · class2/ · class3/ · class4/
├── src/                          # 파이썬 패키지
│   ├── __init__.py
│   ├── config.py                 # 모든 하이퍼파라미터/경로 관리
│   ├── data.py                   # 데이터 로더 & 증강
│   ├── model.py                  # EfficientNet-B0 모델 정의
│   ├── train.py                  # 학습 루프
│   └── evaluate.py               # 평가 & 혼동행렬
├── runs/
│   └── brainMRI_effb0/
|       ├── evaluation/           # confusion matrix
|       ├── train/                 
|       └── validation/
└── README.md                     # ← 지금 보고 있는 파일


| 패키지                    | 버전  | 비고                             |
| ------------------------- | ---- | ---------------------------------|
| Python                    | ≥3.9 | 3.12까지 확인                     |
| TensorFlow                | 2.16 | GPU 4050 Laptop (Cuda 12.1) 기준 |
| Keras-CV                  | ≥0.8 | Random augmentation              |
| NumPy, Pandas, Matplotlib | 최신 |                                  |
