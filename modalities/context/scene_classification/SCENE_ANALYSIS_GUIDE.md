## Scene Classification - Complete Technical Analysis Guide

**Deep technical reference covering architecture, implementation, and interpretation**

---

## Table of Contents

1. [Framework Architecture](#framework-architecture)
2. [Dataset Structure](#dataset-structure)
3. [Stage 1: Baseline Comparison](#stage-1-baseline-comparison)
4. [Stage 2: Hyperparameter Tuning](#stage-2-hyperparameter-tuning)
5. [Stage 3: Report Generation](#stage-3-report-generation)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Performance Analysis](#performance-analysis)
8. [Deployment Guide](#deployment-guide)
9. [Troubleshooting](#troubleshooting)

---

## Framework Architecture

### 3-Stage Design Pattern

```
Input Data → [Stage 1] → [Stage 2] → [Stage 3] → Output Report
```

**Why 3 Stages?**

1. **Stage 1 - Baseline Comparison**: Unbiased model selection
   - Fair comparison with identical conditions
   - Multiple metrics (not just accuracy)
   - Clear winner identification

2. **Stage 2 - Hyperparameter Tuning**: Optimization
   - Systematic search over parameter space
   - Reproducible results
   - Best configuration documentation

3. **Stage 3 - Report Generation**: Justification
   - Professional analysis document
   - Visualizations for stakeholder communication
   - Actionable recommendations

### Dependency Graph

```
Dataset
  ↓
Stage 1: model_comparison.py
  ├─ Inputs: Dataset, config
  ├─ Outputs: comparison_results/baseline_comparison.json
  └─ Determines: Best model for Stage 2
       ↓
Stage 2: hyperparameter_tuning.py
  ├─ Inputs: Dataset, best model, hyperparameter grid
  ├─ Outputs: hyperparameter_tuning/tuning_results.csv
  └─ Determines: Best hyperparameters
       ↓
Stage 3: generate_progress_report.py
  ├─ Inputs: Stage 1 results, Stage 2 results
  ├─ Outputs: markdown report + visualizations
  └─ Determines: Deployment recommendations
```

---

## Dataset Structure

### Directory Layout

```
modalities/context/data/scene/
├── train/                           # Training set (~70% of data)
│   ├── classroom/                   # Scene class 1
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ... (N images)
│   ├── kitchen/                     # Scene class 2
│   │   └── ... (N images)
│   └── office/                      # Scene class 3
│       └── ... (N images)
└── val/                             # Validation set (~30% of data)
    ├── classroom/
    │   └── ... (M images)
    ├── kitchen/
    │   └── ... (M images)
    └── office/
        └── ... (M images)
```

### ImageFolder Format

PyTorch's `ImageFolder` automatically:
- Discovers classes from directory names (classroom=0, kitchen=1, office=2)
- Loads all image formats (JPG, PNG, GIF)
- Creates class labels automatically
- Provides train/val split data loading

```python
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder("data/scene/train")
# Automatically creates:
# - classes: ['classroom', 'kitchen', 'office']
# - class_to_idx: {'classroom': 0, 'kitchen': 1, 'office': 2}
# - samples: list of (path, label) tuples
```

### Image Statistics

Typical Places365 scene images:
- **Format**: JPG, PNG
- **Size**: 224×224 pixels (resized by transforms)
- **Color Space**: RGB
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## Stage 1: Baseline Comparison

### Script: `model_comparison.py`

**Entry Point**: Lines 1-50 (configuration section)

```python
NUM_CLASSES = 3                  # Classroom, Kitchen, Office
IMAGE_SIZE = 224                 # ResNet, EfficientNet standard size
BATCH_SIZE = 32                  # Batch processing size
LR = 1e-4                        # Learning rate (0.0001)
EPOCHS = 15                      # Training epochs
```

### Model Implementations

#### Model 1: MobileNetV3Small

```python
def get_mobilenet_v3_small(num_classes):
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model
```

**Architecture Details**:
- Base: MobileNetV3Small (PyTorch pretrained)
- Input: 3 × 224 × 224
- Stem: Conv2d → Hardswish
- Mobile Inverted Bottleneck Blocks:
  - 1×1 expansion, 3×3 depthwise convolution, 1×1 projection
  - Squeeze-and-Excitation blocks (channel attention)
- Final: Global Average Pooling → Dropout → Linear
- Output: 3 (scenes)

**Parameters**: ~2.5M (smallest)
**Size**: ~10 MB
**Speed**: ~15 ms/image (fastest)

**Why MobileNetV3Small?**
- Designed for mobile/embedded devices
- Extreme parameter efficiency
- Good accuracy-to-speed ratio
- Already proven for scene understanding

#### Model 2: ResNet18

```python
def get_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

**Architecture Details**:
- Input: 3 × 224 × 224
- Stem: Conv2d (7×7) → Batch Norm → ReLU → MaxPool
- Residual Blocks:
  - 4 groups (64, 128, 256, 512 channels)
  - 2 blocks per group
  - Skip connections for gradient flow
- Final: Global Average Pooling → Linear
- Output: 3 (scenes)

**Parameters**: ~11M (medium)
**Size**: ~45 MB
**Speed**: ~25 ms/image (medium)

**Why ResNet18?**
- Reference baseline (well-understood architecture)
- Good balance between model size and accuracy
- Extensively studied for transfer learning
- Standard comparison point

#### Model 3: EfficientNet-B0

```python
def get_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
```

**Architecture Details**:
- Input: 3 × 224 × 224
- Mobile Inverted Bottleneck (MBConv) blocks:
  - Squeeze-and-Excitation for channel attention
  - Stochastic depth for regularization
- Compound scaling: depth×width×resolution = 1.0
- Final: Global Average Pooling → Dropout → Linear
- Output: 3 (scenes)

**Parameters**: ~5.3M (balanced)
**Size**: ~21 MB
**Speed**: ~20 ms/image (fast)

**Why EfficientNet-B0?**
- Better accuracy-to-parameter ratio than ResNet18
- Mobile-friendly architecture (MBConv)
- Good for embedded deployment
- Fills efficiency gap between MobileNet and ResNet

### Training Process

**Baseline Configuration** (same for all 3 models):

```python
baseline_config = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 15,
    'optimizer': 'adam'
}
```

**Training Loop** (Simplified):

```python
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Cosine annealing
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # Calculate metrics
```

**Optimizer**: Adam (default PyTorch params)
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (adaptive learning rate)
- ε = 1e-8 (numerical stability)
- No weight decay (transfers weights as-is)

**Scheduler**: CosineAnnealingLR
- Gradually reduces learning rate over epochs
- Formula: lr_t = lr_0/2 × (1 + cos(π × t/T))
- Helps fine-tune weights at end of training

### Metrics Calculated

**Per-Epoch Metrics**:
```python
accuracy = (predictions == labels).sum() / len(labels)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')
```

**Final Metrics**:
- **Accuracy**: Overall correctness (primary metric)
- **Precision**: Correct positive predictions / total predictions
- **Recall**: Correct positive predictions / actual positives
- **F1-Score**: Harmonic mean of precision and recall

**Performance Metrics**:
```python
model_params = sum(p.numel() for p in model.parameters())
model_size_mb = (param_size + buffer_size) / (1024 * 1024)
inference_times = measure_inference_time(model, device, num_samples=100)
```

### Output: baseline_comparison.json

```json
{
  "MobileNetV3Small": {
    "model_params": 2540000,
    "trainable_params": 2540000,
    "model_size_mb": "10.15",
    "best_acc": "84.50%",
    "best_epoch": 13,
    "training_time_sec": "2450.23",
    "accuracy": "84.50%",
    "precision": "84.30%",
    "recall": "84.50%",
    "f1_score": "84.40%",
    "inference_mean_ms": "15.32",
    "inference_std_ms": "2.15"
  },
  "ResNet18": { ... },
  "EfficientNet-B0": { ... }
}
```

---

## Stage 2: Hyperparameter Tuning

### Script: `hyperparameter_tuning.py`

**Purpose**: Find optimal hyperparameters for best model from Stage 1

### Hyperparameter Grid

```python
learning_rates = [0.001, 0.0005, 0.0001, 0.00005]    # 4 values
batch_sizes = [16, 32, 64]                           # 3 values
optimizers = ['adam', 'sgd']                         # 2 values
total_combinations = 4 × 3 × 2 = 24
```

### Hyperparameter Ranges Explained

#### Learning Rates: [0.001, 0.0005, 0.0001, 0.00005]

Why these ranges?

**Transfer Learning Principle**: Start with base model trained on ImageNet
- If LR too high → Destroy learned features
- If LR too low → Slow convergence

**Rule of Thumb**: Use 0.1× to 10× of baseline
- Baseline: 0.0001
- Our range: 0.00005 to 0.001 ✓ (0.5× to 10×)

**Individual Rates**:
- **0.001** (1000×): May be too aggressive → divergence
- **0.0005** (500×): Moderate fine-tuning
- **0.0001** (1×): Conservative baseline → recommended
- **0.00005** (50×): Very conservative → slow convergence

#### Batch Sizes: [16, 32, 64]

**Gradient Noise Trade-off**:
- **Small batches** (16):
  - Higher gradient noise → regularization effect
  - Better generalization to unseen data
  - Slower per-epoch convergence
  - More frequent weight updates

- **Medium batches** (32):
  - Sweet spot for most tasks
  - Good gradient signal (low noise)
  - Reasonable training speed
  - Stable convergence

- **Large batches** (64):
  - Lower gradient noise
  - Faster per-epoch convergence
  - Risk of overfitting (less regularization)
  - May get stuck in sharp minima

#### Optimizers: ['adam', 'sgd']

**Adam**:
- Adaptive learning rates per parameter
- Good default choice
- Fast convergence
- Forgiving of hyperparameter choices
- Recommended for transfer learning

**SGD**:
- Momentum-based (m = 0.9)
- Stable convergence
- May need careful tuning
- Often generalizes better in production

### Training Procedure

```python
def train_model_tuning(model, train_loader, val_loader, device, config, trial_name):
    """Train model with given hyperparameters"""
    
    # Create optimizer based on config
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    
    # Learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop (15 epochs)
    for epoch in range(config['epochs']):
        # Train phase...
        # Validation phase...
        # Calculate metrics...
    
    return best_accuracy, best_epoch, training_time, metrics
```

### Grid Search Execution

**Trial Loop**:
```
For each learning rate:
  For each batch size:
    For each optimizer:
      Create model (fresh)
      Train 15 epochs
      Record best accuracy
      Save results
```

**Example Output During Search**:
```
Trial 1/24: LR=0.001, BS=16, Opt=adam
  Best Accuracy: 82.1% (Epoch 8)
  F1-Score: 81.8%
  Training Time: 450s

Trial 2/24: LR=0.001, BS=16, Opt=sgd
  Best Accuracy: 80.5% (Epoch 12)
  F1-Score: 80.2%
  Training Time: 420s

... (trials 3-24)

Trial 24/24: LR=0.00005, BS=64, Opt=adam
  Best Accuracy: 85.2% (Epoch 12)  ← Best overall!
  F1-Score: 85.0%
  Training Time: 380s
```

### Output: tuning_results.csv

**Format**: 24 rows × 10 columns

```csv
trial,learning_rate,batch_size,optimizer,best_epoch,accuracy,precision,recall,f1_score,training_time_sec
1,0.001,16,adam,8,82.1,82.0,82.1,81.8,450
2,0.001,16,sgd,12,80.5,80.3,80.5,80.2,420
3,0.001,32,adam,9,83.0,82.9,83.0,82.8,380
4,0.001,32,sgd,11,81.2,81.0,81.2,81.0,360
5,0.001,64,adam,7,82.5,82.4,82.5,82.3,350
6,0.001,64,sgd,10,80.8,80.6,80.8,80.5,330
7,0.0005,16,adam,10,83.5,83.3,83.5,83.2,460
8,0.0005,16,sgd,13,81.8,81.6,81.8,81.5,430
9,0.0005,32,adam,11,84.1,83.9,84.1,83.9,390
10,0.0005,32,sgd,12,82.3,82.1,82.3,82.1,370
11,0.0005,64,adam,9,83.8,83.7,83.8,83.6,360
12,0.0005,64,sgd,11,81.5,81.3,81.5,81.2,340
13,0.0001,16,adam,12,84.3,84.1,84.3,84.1,470
14,0.0001,16,sgd,14,83.0,82.9,83.0,82.8,440
15,0.0001,32,adam,12,84.5,84.3,84.5,84.3,400
16,0.0001,32,sgd,13,83.5,83.3,83.5,83.3,380
17,0.0001,64,adam,11,84.2,84.0,84.2,84.0,370
18,0.0001,64,sgd,12,83.1,83.0,83.1,82.9,350
19,0.00005,16,adam,13,84.8,84.6,84.8,84.6,480
20,0.00005,16,sgd,15,83.2,83.0,83.2,83.0,450
21,0.00005,32,adam,14,85.0,84.8,85.0,84.8,410
22,0.00005,32,sgd,14,83.8,83.6,83.8,83.6,390
23,0.00005,64,adam,12,85.2,85.0,85.2,85.0,380  ← Best!
24,0.00005,64,sgd,13,84.0,83.8,84.0,83.8,360
```

### Output: best_config.json

```json
{
  "learning_rate": 0.00005,
  "batch_size": 64,
  "optimizer": "adam",
  "accuracy": "85.2%",
  "precision": "85.0%",
  "recall": "85.2%",
  "f1_score": "85.0%",
  "best_epoch": 12
}
```

---

## Stage 3: Report Generation

### Script: `generate_progress_report.py`

**Purpose**: Synthesize results into professional markdown report with visualizations

### Report Structure

```
SCENE_CLASSIFICATION_PROGRESS_REPORT.md (48 pages)
├── 1. Executive Summary
├── 2. Baseline Model Comparison
│   ├── Models Selected
│   ├── Results Table
│   ├── Metrics Explanation
│   └── Selection Rationale
├── 3. Technical Specifications
│   ├── Dataset Details
│   ├── Training Configuration
│   ├── Model Architecture
│   └── Data Preprocessing
├── 4. Hyperparameter Tuning
│   ├── Strategy Explanation
│   ├── Top Configurations
│   └── Best Config Justification
├── 5. Deployment Considerations
├── 6. Recommendations
└── Appendix
```

### Key Report Sections

#### Executive Summary
- Summarizes entire analysis
- Key findings (best model, best config)
- Performance metrics
- Recommended actions

#### Model Comparison Analysis
- Describes 3 models evaluated
- Performance table
- Trade-off analysis
- Selection rationale

#### Hyperparameter Tuning
- Explains grid search strategy
- Shows top 5 configurations
- Discusses why best config wins
- Provides deployment guidelines

### Visualizations Generated

#### Visualization 1: model_comparison_visualization.png

**4-panel layout**:

1. **Accuracy Comparison (Bar Chart)**
   - X-axis: Model name
   - Y-axis: Accuracy %
   - Shows which model wins accuracy race

2. **Accuracy vs Model Size (Scatter Plot)**
   - X-axis: Model Size (MB)
   - Y-axis: Accuracy %
   - Shows efficiency trade-off
   - Larger → Better accuracy?
   - Smaller → Faster inference?

3. **Inference Time (Bar Chart)**
   - X-axis: Model name
   - Y-axis: Inference Time (ms)
   - Shows speed comparison
   - Lower is better

4. **Parameter Count (Bar Chart)**
   - X-axis: Model name
   - Y-axis: Number of Parameters (Millions)
   - Shows complexity
   - Fewer → Smaller model size

#### Visualization 2: hyperparameter_tuning_visualization.png

**4-panel layout**:

1. **Learning Rate Impact**
   - X-axis: Learning rate values
   - Y-axis: Average accuracy across other hyperparameters
   - Shows LR sensitivity
   - Which LR works best?

2. **Batch Size Impact**
   - X-axis: Batch size values
   - Y-axis: Average accuracy
   - Shows batch size effect
   - Smaller vs larger batches?

3. **Optimizer Comparison**
   - X-axis: Optimizer type (Adam, SGD)
   - Y-axis: Average accuracy
   - Shows Adam vs SGD performance
   - Error bars show variance

4. **Top 10 Configurations**
   - Horizontal bar chart
   - Shows best 10 combinations found
   - X-axis: Accuracy
   - Y-axis: Config description (LR, BS, Opt)

### Report Generation Code

```python
class ProgressReportGenerator:
    def load_baseline_results(self):
        """Load Stage 1 results"""
        with open("comparison_results/baseline_comparison.json", 'r') as f:
            return json.load(f)
    
    def load_tuning_results(self):
        """Load Stage 2 results"""
        return pd.read_csv("hyperparameter_tuning/tuning_results.csv")
    
    def generate_comparison_visualizations(self, baseline_results):
        """Create 4-panel model comparison chart"""
        # Extract data from results
        # Create 4 subplots
        # Populate with data
        # Save as PNG
    
    def generate_tuning_visualizations(self, tuning_df):
        """Create 4-panel tuning analysis chart"""
        # Extract data from tuning results
        # Create 4 subplots  
        # Populate with data
        # Save as PNG
    
    def generate_full_report(self):
        """Orchestrate all report generation"""
        # Load results
        # Generate markdown report
        # Create visualizations
        # Save all outputs
```

---

## Technical Implementation Details

### Data Augmentation

**Training Transforms**:
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # Standardize size
    transforms.RandomHorizontalFlip(p=0.5),          # 50% flip
    transforms.RandomRotation(10),                   # ±10° rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
    transforms.ToTensor(),                            # Convert to tensor
    transforms.Normalize(                             # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**Validation Transforms** (No augmentation):
```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**Why This Strategy?**
- RandomHorizontalFlip: Scenes look similar when flipped
- RandomRotation: Camera perspective changes
- ColorJitter: Different lighting conditions
- ImageNet Normalization: Matches pretrained weights

### Model Loading and Modification

**Transfer Learning Approach**:
```python
# Load pretrained model
model = models.mobilenet_v3_small(pretrained=True)

# Replace final classification layer
# Original: 1000 classes (ImageNet)
# New: 3 classes (scenes)
model.classifier[3] = nn.Linear(1024, 3)
```

**Why This Works?**
1. Lower layers learn generic features (edges, textures)
2. Middle layers learn mid-level features (patterns)
3. Upper layers learn task-specific features (classes)
4. Only replace final layer for new task

### Evaluation Metrics

**Weighted Metrics** (for imbalanced scenes):
```python
# If classroom has 100 images, kitchen has 50, office has 100
# Weighted metrics give more weight to larger classes

accuracy = (pred == true).sum() / len(labels)

precision_weighted = weighted_average(per_class_precision)
recall_weighted = weighted_average(per_class_recall)
f1_weighted = weighted_average(per_class_f1)
```

---

## Performance Analysis

### Expected Results

**Typical Accuracy Range by Model**:

| Model | Range | Typical |
|-------|-------|---------|
| MobileNetV3Small | 82-86% | 84% |
| ResNet18 | 81-85% | 83% |
| EfficientNet-B0 | 84-88% | 86% |

**After Hyperparameter Tuning**:
- Best configuration typically adds 1-3% accuracy
- Example: 84% → 85.2%

### Inference Speed Benchmarks

**On typical GPU (RTX 3060)**:

| Model | Batch Size | Time | FPS |
|-------|-----------|------|-----|
| MobileNetV3Small | 1 | 15ms | 67 |
| MobileNetV3Small | 32 | 300ms | 107 |
| ResNet18 | 1 | 25ms | 40 |
| EfficientNet-B0 | 1 | 20ms | 50 |

**On CPU**:
- 10-50× slower than GPU
- Not suitable for real-time video processing

### Why Results Vary

**Factors affecting accuracy**:
1. **Dataset composition** - Class imbalance affects metrics
2. **Image quality** - Blurry/low-light images harder to classify
3. **Random seed** - Initialization randomness
4. **Training stochasticity** - Batch order, dropout variations
5. **Hardware differences** - GPU vs CPU, different GPU models

**Reproducibility**:
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## Deployment Guide

### Using Best Configuration

```python
# Load best config from JSON
best_config = {
    'learning_rate': 0.00005,
    'batch_size': 64,
    'optimizer': 'adam'
}

# Train final model
model = get_mobilenet_v3_small(num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=best_config['learning_rate'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Training loop with best config
for epoch in range(15):
    train(model, train_loader, optimizer)
    scheduler.step()
    validate(model, val_loader)

# Save checkpoint
torch.save(model.state_dict(), 'scene_model_best.pth')
```

### Inference Pipeline

```python
class SceneClassifier:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = get_mobilenet_v3_small(3).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, image):
        """Classify single image"""
        # Preprocess
        tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
        
        # Return prediction
        class_idx = probabilities.argmax().item()
        confidence = probabilities[0, class_idx].item()
        return class_idx, confidence
```

### Real-Time Video Processing

```python
import cv2

classifier = SceneClassifier('scene_model_best.pth')
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict scene
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scene_idx, confidence = classifier.predict(pil_image)
    
    # Display result
    scenes = ['Classroom', 'Kitchen', 'Office']
    label = f"{scenes[scene_idx]} ({confidence:.1%})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Scene Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "CUDA out of memory"

**Cause**: Batch too large for GPU VRAM

**Solutions**:
```python
# Option 1: Reduce batch size
batch_sizes = [16, 32]  # Remove 64

# Option 2: Clear GPU memory
torch.cuda.empty_cache()

# Option 3: Use gradient accumulation
for micro_batch in split_batch(batch):
    output = model(micro_batch)
    loss = criterion(output, labels)
    (loss / num_micro_batches).backward()
    accumulation_counter += 1
    if accumulation_counter == num_micro_batches:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. "Dataset not found"

**Cause**: Wrong data path

**Check**:
```bash
ls modalities/context/data/scene/train/classroom/
# Should show image files

ls modalities/context/data/scene/val/
# Should show subdirectories
```

**Fix**:
```python
# Use absolute path
dataset_base = Path("/full/path/to/adaptive-multimodal-hri/modalities/context/data/scene")
train_dir = dataset_base / "train"
```

#### 3. "Low accuracy (< 70%)"

**Cause**: Model not converging

**Check**:
- [ ] Data loading correctly? (verify image shapes)
- [ ] Learning rate appropriate? (check loss curve)
- [ ] Enough epochs? (increase to 30)
- [ ] Data quality? (check for corrupted images)

**Solutions**:
```python
# Increase training epochs
EPOCHS = 30

# Visualize training loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

#### 4. "Training very slow"

**Cause**: Running on CPU

**Verify GPU usage**:
```python
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# In separate terminal:
nvidia-smi  # Check GPU utilization
```

**Force GPU**:
```python
device = torch.device("cuda:0")
model = model.to(device)
```

---

## Appendix: Reference Materials

### Key References

1. **MobileNetV3 Paper**: "Searching for MobileNetV3"
   - Efficiency-optimized architecture
   - Squeeze-and-Excitation blocks

2. **ResNet Paper**: "Deep Residual Learning for Image Recognition"
   - Skip connections enable very deep networks
   - Reference architecture

3. **EfficientNet Paper**: "EfficientNet: Rethinking Model Scaling"
   - Compound scaling of depth, width, resolution
   - Mobile-friendly design

4. **ImageNet Normalization**
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
   - Used across all transfer learning models

### Important Classes and Functions

**PyTorch ImageFolder**:
- `torchvision.datasets.ImageFolder(root, transform)` - Auto-discover dataset
- Requires: root/class_name/image.ext structure

**Models from torchvision.models**:
- `models.mobilenet_v3_small(pretrained=True)`
- `models.resnet18(pretrained=True)`
- `models.efficientnet_b0(pretrained=True)`

**Metrics from sklearn.metrics**:
- `accuracy_score(y_true, y_pred)`
- `precision_score(y_true, y_pred, average='weighted')`
- `recall_score(y_true, y_pred, average='weighted')`
- `f1_score(y_true, y_pred, average='weighted')`

---

**Previous**: [Framework Overview](SCENE_FRAMEWORK_OVERVIEW.md)  
**Reference**: [Execution Checklist](SCENE_EXECUTION_CHECKLIST.md)
