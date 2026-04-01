# Brain MRI 종양 분류 상세 가이드 (Technical Details)

## 1. 하이퍼파라미터 설정 (`src/config.py`)
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `img_height` | 224 | 모델 입력 이미지 높이 |
| `img_width` | 224 | 모델 입력 이미지 너비 |
| `batch_size` | 32 | GPU 메모리가 부족할 경우 16으로 조정 |
| `split.train` | 0.7 | 전체 데이터셋 중 학습용 비율 |
| `split.val` | 0.3 | 전체 데이터셋 중 검증용 비율 |
| `mixed_precision` | True | float16 연산을 활성화하여 VRAM 절약 및 속도 최적화 |

### 2단계(2-Stage) 학습 스케줄러
- **Stage 1 (분류 헤드 안정화)**: 백본 모델 레이어 완전 고정. `epochs = 30`, `learning_rate = 1e-3`
- **Stage 2 (Fine-tuning)**: `unfreeze_layers = 20`. 상위 20개 레이어만 가중치를 업데이트함. `epochs = 30`, `learning_rate = 1e-4` (과적합 방지를 위해 낮은 lr 사용)

## 2. 데이터 전처리 및 증강 (`src/data_loader.py`)
의료 영상 데이터셋 특성상 샘플 수가 부족할 수 있습니다. 이를 보완하고 모델의 일반화(Generalization) 성능을 높이기 위해 다음 레이어 증강 기법이 적용됩니다:
- `tf.keras.layers.RandomFlip("horizontal")`: 수평 뒤집기
- `tf.keras.layers.RandomRotation(0.1)`: -10% ~ +10% 각도 회전
- `tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1))`: 90% ~ 110% 스케일링 확대/축소
- `tf.keras.layers.RandomContrast(0.2)`: 대비 변환

## 3. EfficientNet-B0 아키텍처 커스텀
**기본 모델**: ImageNet 가중치 기반 `EfficientNetB0` (`include_top=False`)
**Custom Classification Head**:
1. `GlobalAveragePooling2D()`: 7x7 공간 차원을 제거하고 1280차원 벡터를 추출
2. `Dropout(0.3)`: 과적합 방지를 위한 정규화 드롭아웃
3. `Dense(4, activation='softmax')`: 4개 종양 클래스로 확률 분류

## 4. 트러블슈팅 및 튜닝 가이드
- **GPU Out of Memory (OOM) 발생 시**: `config.py`의 `batch_size`를 16 또는 8로 줄입니다. `mixed_precision=True`가 되어 있는지 확인하세요.
- **검증 정확도가 정체되거나 낮을 때 (Underfitting/Overfitting)**:
  - 데이터 증강 강도 상향 (`RandomRotation(0.2)` 등으로 조정).
  - Stage 2에서 언프리즈(Unfreeze)하는 레이어 개수를 늘립니다 (`unfreeze_layers = 50`).
- **새로운 종양 질환 클래스를 추가하고 싶을 때**:
  - `multi_class_dataset` 내에 새 클래스명으로 폴더를 추가하고 이미지 데이터를 넣습니다.
  - `src/model.py`의 `build_model(num_classes=4)` 인자를 새로운 클래스 개수(예: 5)로 변경한 후 다시 훈련을 돌립니다.
