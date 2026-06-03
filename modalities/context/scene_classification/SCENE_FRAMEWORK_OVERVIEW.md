## Scene Classification Framework Overview

**Complete architecture and design explanation for the 3-stage analysis framework**

---

## 📐 Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│         SCENE CLASSIFICATION ANALYSIS FRAMEWORK                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   STAGE 1    │    │   STAGE 2    │    │   STAGE 3    │     │
│  │ Baseline     │───▶│ Hyperparameter│───▶│   Report     │     │
│  │ Comparison   │    │ Tuning        │    │ Generation   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│       Input              Input              Input               │
│   • 3 Models         • Best model       • Stage 1 results    │
│   • Same config      • 24 configs       • Stage 2 results    │
│   • Full dataset     • Full dataset     • Metrics            │
│                                                                 │
│       Output             Output             Output              │
│   • Metrics          • Best config      • MD Report          │
│   • JSON file        • CSV file         • Visualizations     │
│   • Best model       • Statistics       • JSON metrics       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Three-Stage Pipeline

### Stage 1: Baseline Model Comparison

**Purpose**: Objectively compare three lightweight models using identical hyperparameters.

**Models Tested**:
1. **MobileNetV3Small** (Current)
   - Parameters: ~2.5M
   - Size: ~10 MB
   - Speed: Fastest (~15 ms)
   - Design: Lightweight for embedded systems

2. **ResNet18** (Balanced)
   - Parameters: ~11M
   - Size: ~45 MB
   - Speed: Medium (~25 ms)
   - Design: Reference baseline

3. **EfficientNet-B0** (Efficient)
   - Parameters: ~5.3M
   - Size: ~21 MB
   - Speed: Fast (~20 ms)
   - Design: Optimized accuracy-to-efficiency ratio

**Baseline Configuration** (same for all 3 models):
```
Learning Rate: 0.0001
Batch Size: 32
Optimizer: Adam
Epochs: 15
Loss: CrossEntropyLoss
Scheduler: CosineAnnealingLR
```

**Measurements**:
- ✓ Training accuracy
- ✓ Validation accuracy (per class)
- ✓ Precision, Recall, F1-Score
- ✓ Model parameters count
- ✓ Model size (MB)
- ✓ Inference time (ms)
- ✓ Training time (seconds)

**Output**:
```json
{
  "MobileNetV3Small": {
    "model_params": 2540000,
    "model_size_mb": "10.2",
    "accuracy": "84.5%",
    "precision": "84.2%",
    "f1_score": "84.1%",
    "inference_mean_ms": "15.3"
  }
  // ... etc for other models
}
```

---

### Stage 2: Hyperparameter Tuning

**Purpose**: Systematically optimize training configuration for best-performing model from Stage 1.

**Hyperparameter Grid**:
```
Learning Rates:    [0.001, 0.0005, 0.0001, 0.00005]  (4 values)
Batch Sizes:       [16, 32, 64]                       (3 values)
Optimizers:        ['adam', 'sgd']                    (2 values)
────────────────────────────────────────────────────────
Total Combinations: 4 × 3 × 2 = 24 configurations
```

**Why These Ranges?**

*Learning Rates*:
- 0.001: Too aggressive, may diverge
- 0.0005: Moderate fine-tuning
- 0.0001: Conservative (recommended for transfer learning)
- 0.00005: Very conservative, slow convergence

*Batch Sizes*:
- 16: Small, high gradient noise, better generalization
- 32: Medium (sweet spot), good stability
- 64: Large, faster training, less noise

*Optimizers*:
- Adam: Adaptive learning rates, good defaults
- SGD: Momentum-based, requires careful tuning

**Example Tuning Job**:
```
Trial 1: LR=0.001, BS=16, Opt=adam → Accuracy=82.1%
Trial 2: LR=0.001, BS=16, Opt=sgd  → Accuracy=80.5%
Trial 3: LR=0.001, BS=32, Opt=adam → Accuracy=83.0%
...
Trial 24: LR=0.00005, BS=64, Opt=adam → Accuracy=85.2% ← Best
```

**Output**:
```csv
trial,learning_rate,batch_size,optimizer,accuracy,f1_score,training_time_sec
1,0.001,16,adam,82.1,81.8,450
2,0.001,16,sgd,80.5,80.2,420
...
24,0.00005,64,adam,85.2,85.0,380
```

**Best Configuration**:
```json
{
  "learning_rate": 0.00005,
  "batch_size": 64,
  "optimizer": "adam",
  "accuracy": "85.2%",
  "f1_score": "85.0%",
  "best_epoch": 12
}
```

---

### Stage 3: Progress Report Generation

**Purpose**: Synthesize results from Stages 1 & 2 into professional report with visualizations.

**Report Contents**:

1. **Executive Summary**
   - Key findings
   - Selected model and configuration
   - Performance metrics

2. **Baseline Comparison Analysis**
   - Model descriptions
   - Performance table
   - Selection rationale

3. **Technical Specifications**
   - Dataset details
   - Training configuration
   - Model architecture
   - Data preprocessing

4. **Hyperparameter Tuning Analysis**
   - Grid search strategy
   - Top 5 configurations
   - Best configuration details

5. **Deployment Considerations**
   - Real-time processing capability
   - Resource requirements
   - Integration points

6. **Recommendations**
   - Model selection justification
   - Next steps
   - Future improvements

**Visualizations Generated**:

1. **model_comparison_visualization.png** (4-panel):
   - Accuracy comparison (bar chart)
   - Accuracy vs Model Size (scatter)
   - Inference Time (bar chart)
   - Parameter Count (bar chart)

2. **hyperparameter_tuning_visualization.png** (4-panel):
   - Learning Rate Impact
   - Batch Size Impact
   - Optimizer Comparison
   - Top 10 Configurations

**Report Output**:
```
progress_report/
├── SCENE_CLASSIFICATION_PROGRESS_REPORT.md (48 pages)
├── model_comparison_visualization.png
├── hyperparameter_tuning_visualization.png
├── best_configuration.json
└── detailed_tuning_results.csv
```

---

## 🔄 Data Flow

```
Dataset
├── Train Images (Classroom, Kitchen, Office)
└── Val Images (Classroom, Kitchen, Office)
           ↓
    Data Preprocessing
├── Resize to 224×224
├── Augmentation (rotation, flip, color)
└── Normalization (ImageNet stats)
           ↓
    ┌─────────────┬──────────────┬────────────────┐
    ↓             ↓              ↓                ↓
Stage 1       Model 1       Model 2           Model 3
Baseline     MobileV3      ResNet18       EfficientNet-B0
Comparison        ↓             ↓                ↓
    ├────────────┼──────────────┼────────────────┤
    │ Train × 15 epochs with same config       │
    │ Measure: accuracy, F1, inference time    │
    ├────────────┬──────────────┬────────────────┤
    ↓            ↓              ↓                ↓
Results:    Acc=84.5%    Acc=83.5%          Acc=86.1%
         Best Model: EfficientNet-B0 (or MobileV3, etc)
                         ↓
                    Stage 2
                Hyperparameter
                    Tuning
            (24 configurations tested)
                         ↓
                  Best Config Found
              LR=0.00005, BS=64, adam
                         ↓
                    Stage 3
                    Report
                Generation
                         ↓
         Professional Analysis Report
         + Visualizations + Metrics
```

---

## 🎯 Design Principles

### 1. **Comprehensive Comparison**
- Three models covering efficiency spectrum
- Identical baseline conditions for fairness
- Multiple performance metrics (not just accuracy)

### 2. **Systematic Optimization**
- Grid search covers reasonable hyperparameter ranges
- Deterministic and reproducible
- Results fully documented

### 3. **Professional Documentation**
- Automated report generation
- Reproducible visualizations
- Deployment-ready analysis

### 4. **Transfer Learning Focus**
- All models use ImageNet pretraining
- Conservative learning rates for fine-tuning
- Practical for real robot deployment

### 5. **Scene Classification Specific**
- 3 scene classes (Classroom, Kitchen, Office)
- Places365 dataset adapted
- Real-time inference requirements

---

## 📊 Key Design Decisions

### Why These Three Models?

| Model | Reason |
|-------|--------|
| **MobileNetV3Small** | Current model, ultra-lightweight, embedded deployment |
| **ResNet18** | Reference point, well-studied, good baseline |
| **EfficientNet-B0** | Better accuracy-to-parameter ratio, mobile-friendly |

**Not included**:
- ❌ Larger models (ResNet50, VGG) - Too heavy for real-time
- ❌ Vision Transformers - Too compute-intensive for robots
- ❌ Specialized models - Framework is general-purpose

### Why This Grid Search?

| Aspect | Reason |
|--------|--------|
| 4 LRs | Covers typical transfer learning range (0.1× to 10× baseline) |
| 3 BSs | Covers efficiency vs. stability trade-off |
| 2 Opts | Covers adaptive vs. momentum-based approaches |
| 24 Total | Manageable time (~2 hours), covers design space |

**Not 100+ combinations**: Would take 8+ hours, diminishing returns after 24

### Why 15 Epochs?

- Balances convergence with computation time
- Early stopping may trigger at epoch 10-12
- Long enough for reliable estimates
- Short enough for feasible tuning

---

## 🔧 Customization Points

Framework is modular and extensible:

```python
# Stage 1: Add more models
def get_your_model(num_classes):
    return YourModel()

models_to_compare['YourModel'] = get_your_model

# Stage 2: Modify grid search
learning_rates = [0.01, 0.001, 0.0001]  # Change this
batch_sizes = [32, 64]                   # Change this

# Stage 3: Update report template
report += "## Your Custom Section\n..."  # Add sections
```

---

## 📈 Expected Results

Typical performance on Places365 Custom Dataset (3 scenes):

| Model | Accuracy | F1 | Inference | Params |
|-------|----------|-----|-----------|--------|
| MobileNetV3Small | 82-86% | 82-86% | 12-18ms | 2.5M |
| ResNet18 | 81-85% | 81-85% | 20-30ms | 11M |
| EfficientNet-B0 | 84-88% | 84-88% | 18-25ms | 5.3M |

After tuning, best config typically improves by 1-3% accuracy.

---

## 🚀 Deployment Path

```
1. Run Complete Analysis
   └─ Identify best model + config

2. Train Final Model
   └─ Use best_config hyperparameters
   └─ Train on full training set

3. Evaluate on Test Set
   └─ Measure performance metrics
   └─ Check real-world images

4. Deploy to Robot
   └─ Export model weights
   └─ Integrate with emotion/gesture
   └─ Real-time inference loop

5. Monitor Performance
   └─ Track accuracy over time
   └─ Adapt if distribution shifts
   └─ Retrain if needed
```

---

## 🔗 Integration Points

Scene classification integrates with:

**Emotion Recognition**:
```
Scene Context → Adjust emotion thresholds per environment
Kitchen → Higher satisfaction baseline
Office → Higher stress expected
```

**Gesture Recognition**:
```
Scene Context → Use scene-specific gesture models
Classroom → Academic gestures
Kitchen → Food preparation gestures
```

**Robot Behavior**:
```
Scene Context → Modify interaction strategy
Public environment → Formal mode
Private environment → Casual mode
```

---

## ✅ Validation Checklist

After running complete analysis:

- [ ] Stage 1: `comparison_results/baseline_comparison.json` exists
- [ ] Stage 2: `hyperparameter_tuning/tuning_results.csv` has 24 rows
- [ ] Stage 3: `progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md` generated
- [ ] Visualizations: 2 PNG files created
- [ ] Best config: identified and saved in JSON

---

## 📞 Support

**Questions about framework?**
- See [SCENE_ANALYSIS_GUIDE.md](SCENE_ANALYSIS_GUIDE.md) for technical details
- Check [START_HERE.md](START_HERE.md) for quick overview
- Review [SCENE_EXECUTION_CHECKLIST.md](SCENE_EXECUTION_CHECKLIST.md) for step-by-step

---

**Next**: Run the analysis with [SCENE_QUICK_START.md](SCENE_QUICK_START.md)
