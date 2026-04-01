# Brain MRI Classification Technical Details

## 1. Configuration Parameters (`src/config.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_height` | 224 | Model input height |
| `img_width` | 224 | Model input width |
| `batch_size` | 32 | Reduce to 16 or 8 if you encounter OOM errors |
| `split.train` | 0.7 | Ratio for training set |
| `split.val` | 0.3 | Ratio for validation set |
| `mixed_precision` | True | Uses float16 for massive memory savings & speedup |

### 2-Stage Training Schedule
- **Stage 1 (Head Stabilization)**: Frozen base layers. `epochs = 30`, `lr = 1e-3`.
- **Stage 2 (Fine-tuning)**: Unfreezes top layers (`unfreeze_layers = 20`). `epochs = 30`, `lr = 1e-4` (Lower LR protects pre-trained representations).

## 2. Data Augmentation (`src/data_loader.py`)
Medical image datasets are often small. To improve robustness and generalization, the pipeline applies these dynamic transformations:
- `RandomFlip("horizontal")`
- `RandomRotation(0.1)`: Rotates by up to ±10%
- `RandomZoom((-0.1, 0.1))`: Zooms in/out 90-110%
- `RandomContrast(0.2)`

## 3. Custom Model Architecture
- **Backbone**: EfficientNet-B0 pre-trained on ImageNet (`include_top=False`).
- **Classification Head**:
  1. `GlobalAveragePooling2D`: Converts the final 7x7 spatial tensor into a 1280-dim vector.
  2. `Dropout(0.3)`: Regularization layer.
  3. `Dense(4, activation='softmax')`: Outputs probability distributions for the 4 classes.

## 4. Tuning Guide
- **Handling OOM**: Decrease `batch_size`.
- **Improving Accuracy**: Try increasing `unfreeze_layers` to 50 in Stage 2 to let the network adapt deeper layers to medical textures.
- **Adding Classes**: Create a new folder in `multi_class_dataset`, add the images, and change `num_classes=5` inside `src/model.py`. The data loader automatically picks up the new structure.
