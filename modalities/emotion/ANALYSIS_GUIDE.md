# Emotion Recognition Model - Complete Analysis & Deployment Guide

## Overview

This directory contains a comprehensive framework for:
1. **Model Comparison**: Evaluating multiple lightweight pretrained models
2. **Hyperparameter Tuning**: Optimizing training configuration
3. **Progress Report Generation**: Creating detailed analysis for deployment

**Target Deployment**: NVIDIA Jetson Orin Nano (8GB LPDDR5 RAM, limited compute)
**Dataset**: RAF-DB (Real-world Affective Faces Database)
**Framework**: PyTorch with torchvision

---

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Navigate to emotion directory
cd modalities/emotion

# Run complete analysis (all 3 stages)
python run_complete_analysis.py
```

**Estimated Time**: 2-4 hours (GPU dependent)

### Option 2: Run Individual Stages

```bash
# Stage 1: Baseline Comparison (30-60 minutes)
python model_comparison.py

# Stage 2: Hyperparameter Tuning (60-120 minutes)  
python hyperparameter_tuning.py

# Stage 3: Report Generation (2-5 minutes)
python generate_progress_report.py
```

---

## What Each Script Does

### 1. `model_comparison.py` - Baseline Model Comparison

**Purpose**: Compare three lightweight models with identical configurations

**Models Evaluated**:
- **MobileNetV2** (3.5M parameters, ~14 MB)
  - Optimized for mobile/edge devices
  - Fastest inference
  - Smallest model size
  
- **EfficientNet-B0** (5.3M parameters, ~20 MB)
  - Better accuracy-to-size ratio
  - Slightly heavier, slightly better accuracy
  
- **ResNet18** (11.2M parameters, ~44 MB)
  - Reference baseline
  - Heavier, better accuracy but not edge-friendly

**Baseline Configuration**:
```
Learning Rate: 0.0001
Batch Size: 32
Optimizer: Adam
Epochs: 25
```

**Outputs**:
- `comparison_results/baseline_comparison.json` - Detailed metrics
- Model statistics (parameters, size, inference time)
- Performance metrics (accuracy, precision, recall, F1)

**Key Metrics Recorded**:
```json
{
  "MobileNetV2": {
    "model_params": 3504872,
    "model_size_mb": "13.45",
    "accuracy": "84.03%",
    "f1_score": "83.77%",
    "inference_mean_ms": "24.53"
  }
}
```

---

### 2. `hyperparameter_tuning.py` - Grid Search Optimization

**Purpose**: Find optimal hyperparameters for MobileNetV2

**Hyperparameter Grid**:
```
Learning Rates:    [0.001, 0.0005, 0.0001, 0.00005]
Batch Sizes:       [16, 32, 64]
Optimizers:        ['adam', 'sgd']
Total Combinations: 24
```

**Why These Values?**

#### Learning Rate Justification
- `0.001`: Too aggressive for transfer learning, may overshoot
- `0.0005`: Medium-high, good for some tasks
- `0.0001`: **RECOMMENDED** - Stable fine-tuning of pretrained weights
- `0.00005`: Very conservative, may be too slow

Transfer learning requires lower LR because pretrained weights are already good.

#### Batch Size Justification
- `16`: Higher gradient noise (acts as regularization), but slower training
- `32`: **RECOMMENDED** - Best GPU utilization, stable convergence
- `64`: Faster training, but may overfit, requires more careful tuning

#### Optimizer Justification
- `Adam`: Adaptive per-parameter learning rates, works well for transfer learning
- `SGD`: More stable in some cases, but requires careful LR tuning

**Outputs**:
- `hyperparameter_tuning/tuning_results.csv` - All 24 results
- `hyperparameter_tuning/best_config.json` - Optimal configuration
- Console prints top 5 configurations

**Expected Output**:
```csv
trial,learning_rate,batch_size,optimizer,best_epoch,accuracy,precision,recall,f1_score,training_time_sec
1,0.001,16,adam,15,0.8145,0.8134,0.8145,0.8124,450.23
...
24,0.00005,64,sgd,22,0.8390,0.8385,0.8390,0.8370,380.45
```

---

### 3. `generate_progress_report.py` - Report Generation

**Purpose**: Compile results into comprehensive progress report

**Inputs**:
- Baseline comparison results
- Hyperparameter tuning results
- Best configuration found

**Outputs**:
```
progress_report/
├── EMOTION_MODEL_PROGRESS_REPORT.md          (Main report)
├── model_comparison_visualization.png         (4-panel comparison)
├── hyperparameter_tuning_visualization.png   (Tuning analysis)
├── detailed_tuning_results.csv               (Full results)
└── best_configuration.json                   (Optimal config)
```

**Report Contents**:
1. Executive Summary
2. Baseline Model Comparison Analysis
3. Model Selection Justification
4. Hyperparameter Tuning Strategy
5. Technical Specifications
6. Performance Analysis by Emotion Class
7. Jetson Orin Nano Deployment Guide
8. Recommendations & Next Steps

---

## Dataset Information

### RAF-DB Structure
```
data/
├── train/
│   ├── 1/ (Surprise)    - ~1,400 images
│   ├── 2/ (Fear)        - ~350 images
│   ├── 3/ (Disgust)     - ~700 images
│   ├── 4/ (Happy)       - ~7,000 images
│   ├── 5/ (Sad)         - ~2,000 images
│   ├── 6/ (Anger)       - ~800 images
│   └── 7/ (Neutral)     - ~400 images
│   Total: ~12,271 images
│
└── test/
    ├── 1/ - ~330 images
    ├── 2/ - ~75 images
    ├── 3/ - ~160 images
    ├── 4/ - ~1,185 images
    ├── 5/ - ~480 images
    ├── 6/ - ~165 images
    └── 7/ - ~680 images
    Total: ~3,068 images

Image Format: 224×224 RGB
Preprocessing: ImageNet normalization
```

### Class Imbalance
- **Happy**: 1,185 test samples (overrepresented)
- **Neutral**: 680 test samples
- **Sad**: 478 test samples
- **Surprise**: 329 test samples
- **Disgust**: 160 test samples
- **Anger**: 162 test samples
- **Fear**: 74 test samples (underrepresented)

**Impact**: Model performs best on Happy, may struggle with Fear

---

## Configuration Files

### `config.py` - Training Hyperparameters

```python
NUM_CLASSES = 7           # Number of emotions
IMAGE_SIZE = 224          # Input image size
BATCH_SIZE = 32           # Training batch size
LR = 0.0001              # Learning rate
EPOCHS = 25              # Training epochs

EMOTION_LABELS = [
    "Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"
]
```

**How to Modify**:
- Adjust `BATCH_SIZE` based on GPU memory (decrease if OOM)
- Adjust `LR` based on tuning results
- Increase `EPOCHS` if validation accuracy still improving at epoch 25

### `utils/transforms.py` - Data Augmentation

```python
# Training augmentation (data augmentation)
- Random rotation: ±10°
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

# Test augmentation (no augmentation)
- Only ImageNet normalization
```

---

## Model Architecture Details

### MobileNetV2 Architecture (Selected)

```
MobileNetV2 = Lightweight Convolutional Neural Network
├── Input: 224×224×3
├── Stem: Conv2d (3→32) + BatchNorm + ReLU
├── 16 Inverted Residual Blocks:
│   ├── Expansion: Channel increase (1/6 ratio)
│   ├── DepthWise Conv: 3×3 spatial convolution
│   ├── Projection: Channel decrease
│   └── Bottleneck structure
├── Final Conv: Expand to 1280 channels
├── Global Average Pooling: 1280 → 1 per channel
├── Dropout: 0.2
└── Classifier: Dense 1280→7 (7 emotions)

Key Features:
- Depthwise separable convolutions (8-9x fewer params)
- Bottleneck residual blocks
- Linear bottlenecks (no activation after projection)
- Total params: 3.5M
- Model size: ~13-14 MB
```

### Why MobileNetV2 for Jetson Orin Nano?

```
Jetson Orin Nano Constraints:
├── GPU Memory: 8GB shared (not just GPU)
├── Inference Power: 12 TFLOPS (limited)
├── Peak Bandwidth: 102 GB/s
└── Typical Power: 5-15W

MobileNetV2 Advantages:
✓ 3.5M params vs ResNet18's 11.2M (68% reduction)
✓ ~13 MB vs ResNet50's ~100MB (87% reduction)
✓ Inference: 20-30ms (30-50 FPS video capability)
✓ Power efficient: 2-3W GPU portion
✓ Proven on mobile devices (iPhone, Android)
```

---

## Expected Results

### Performance Range

Based on RAF-DB literature and typical transfer learning:

```
Baseline Results (All models, same config):
┌──────────────────┬───────────┬──────────────┬─────────────┐
│ Model            │ Accuracy  │ Model Size   │ Inf. Time   │
├──────────────────┼───────────┼──────────────┼─────────────┤
│ MobileNetV2      │ ~84%      │ ~13 MB       │ ~25 ms      │
│ EfficientNet-B0  │ ~85%      │ ~20 MB       │ ~30 ms      │
│ ResNet18         │ ~82%      │ ~44 MB       │ ~35 ms      │
└──────────────────┴───────────┴──────────────┴─────────────┘

Tuning Results:
- Best accuracy: ~84.5-85% possible
- Typical improvement: 0.5-1% from baseline
- Best config likely: LR=0.0001 or 0.0005, BS=16 or 32, Adam
```

### Per-Class Performance Expected

```
Happy:    ~93-95% (distinctive features)
Neutral:  ~83-86% (baseline)
Sad:      ~80-83% (clear downturned mouth)
Anger:    ~75-78% (confuses with disgust)
Surprise: ~78-82% (confuses with fear)
Disgust:  ~55-65% (very subtle, confuses with anger)
Fear:     ~60-70% (rare in dataset, confuses with surprise)

Overall Accuracy: ~83-85%
```

---

## Deployment Workflow

### Step 1: Review Results

```bash
# Main report
cat progress_report/EMOTION_MODEL_PROGRESS_REPORT.md

# View visualizations
# Open: progress_report/model_comparison_visualization.png
# Open: progress_report/hyperparameter_tuning_visualization.png
```

### Step 2: Use Best Configuration

```bash
# From best_config.json:
LR = [tuned_learning_rate]
BATCH_SIZE = [tuned_batch_size]
OPTIMIZER = [tuned_optimizer]

# Update config.py with these values
# Retrain final model:
python train.py
```

### Step 3: Convert for Edge Deployment

```bash
# Convert to ONNX (Jetson Orin Nano friendly)
python export_to_onnx.py

# Or use TensorRT optimization
python export_to_tensorrt.py

# Result: 2-4x faster inference on Jetson
```

### Step 4: Benchmark on Jetson

```bash
# On Jetson Orin Nano:
ssh jetson@[ip]
python benchmark_on_jetson.py

# Expected: 20-50 FPS video processing
```

---

## Troubleshooting

### Out of Memory (OOM) Error

**Solution**: Reduce batch size
```python
# config.py
BATCH_SIZE = 16  # Instead of 32
```

### Training Too Slow

**Possible Causes**:
- Batch size too small (more iterations)
- Learning rate too low (not converging)
- Too many workers in DataLoader

**Solution**:
```python
BATCH_SIZE = 64  # Increase batch size
# In model_comparison.py, set num_workers > 0
```

### Model Not Converging

**Possible Causes**:
- Learning rate too high (diverging)
- Learning rate too low (stuck)
- Data not loading correctly

**Solution**:
```python
LR = 0.0001      # Use middle value from tuning
# Verify data loading with print statements
```

### Inference Time Too Slow

**On GPU**:
- May need more VRAM
- Try FP16 precision instead of FP32

**On Jetson**:
- Use TensorRT optimization (2-4x speedup)
- Use INT8 quantization (4x speedup, ~1% accuracy loss)

---

## Advanced: Model Optimization for Jetson

### FP16 (Half Precision) Inference
```bash
# 2x faster, minimal accuracy loss
python export_fp16_model.py
```

### INT8 Quantization
```bash
# 4x smaller, 4x faster, ~1% accuracy loss
python quantize_to_int8.py
```

### Pruning
```bash
# Remove 30-50% of parameters
python prune_model.py
```

### Knowledge Distillation
```bash
# Distill large teacher → small student model
python distill_model.py
```

---

## File Structure

```
modalities/emotion/
├── data/                          # RAF-DB dataset
│   ├── train/                     # Training images
│   └── test/                      # Testing images
│
├── models/
│   └── mobilenet_emotion.py       # Model definition
│
├── utils/
│   └── transforms.py              # Data augmentation
│
├── config.py                       # Training config
├── train.py                        # Training script
├── evaluate_model.py               # Evaluation script
│
├── model_comparison.py             # Stage 1: Baseline comparison
├── hyperparameter_tuning.py        # Stage 2: HP tuning
├── generate_progress_report.py     # Stage 3: Report generation
├── run_complete_analysis.py        # Master script
│
├── comparison_results/             # Stage 1 outputs
├── hyperparameter_tuning/          # Stage 2 outputs
└── progress_report/                # Stage 3 outputs
    ├── EMOTION_MODEL_PROGRESS_REPORT.md
    ├── model_comparison_visualization.png
    └── hyperparameter_tuning_visualization.png
```

---

## Next Steps After Analysis

1. **Review Report**
   - Read EMOTION_MODEL_PROGRESS_REPORT.md
   - Understand model selection rationale
   - Review deployment considerations

2. **Retrain with Best Config**
   - Use optimal hyperparameters from tuning
   - Save best checkpoint
   - Validate on held-out test set

3. **Prepare for Deployment**
   - Convert model to ONNX/TensorRT
   - Benchmark on actual Jetson hardware
   - Test real-time video processing

4. **Integration**
   - Integrate with scene classification (context modality)
   - Integrate with gesture recognition (gesture modality)
   - Build multimodal emotion recognition system

5. **Optimization**
   - Consider ensemble methods for robustness
   - Implement confidence-based fallbacks
   - Add contextual weighting from other modalities

---

## References

### Models
- **MobileNetV2**: Sandler et al., 2018. "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- **EfficientNet**: Tan & Le, 2019. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **ResNet**: He et al., 2015. "Deep Residual Learning for Image Recognition"

### Emotion Recognition
- **RAF-DB**: Li et al., 2019. "Real-world Affective Faces Database"
- **FER Survey**: Khaireddin & Chen, 2016. "Facial Expression Recognition: Review of Trends and Common Approaches"

### Edge Deployment
- **TensorRT**: NVIDIA TensorRT Inference Server Optimization
- **Jetson Orin**: NVIDIA Jetson Orin Nano Technical Brief

---

## Support & Issues

For questions or issues:
1. Check this README
2. Review error messages in console output
3. Check generated report for deployment guidance
4. Refer to model documentation

---

*Last Updated: 2024*
*Emotion Recognition System for Adaptive Human-Robot Interaction*
