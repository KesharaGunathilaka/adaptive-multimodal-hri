## Scene Classification - Quick Start Guide

**TL;DR**: Run 3 commands to analyze scene classification models

---

## 🚀 Three Steps to Complete Analysis

### Step 1: Open Terminal
```bash
cd d:\Documents\Project\12. Adaptive Human Robot Interaction\adaptive-multimodal-hri\modalities\context\scene_classification\
```

### Step 2: Run Master Script
```bash
python run_complete_analysis.py
```

The script will:
- ✓ Run Stage 1: Compare MobileNetV3Small, ResNet18, EfficientNet-B0
- ✓ Run Stage 2: Test 24 hyperparameter combinations
- ✓ Run Stage 3: Generate comprehensive report with visualizations

Estimated time: **2-4 hours** (depends on GPU)

### Step 3: Review Results
```bash
# View generated report
cat progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md

# Or open with text editor:
# - Windows: Start progress_report\SCENE_CLASSIFICATION_PROGRESS_REPORT.md
# - VS Code: code progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md
```

---

## 📊 What You'll Get

**JSON Results** (baseline_comparison.json):
```json
{
  "MobileNetV3Small": {"accuracy": "85.2%", "f1_score": "85.0%", ...},
  "ResNet18": {"accuracy": "83.5%", "f1_score": "83.2%", ...},
  "EfficientNet-B0": {"accuracy": "86.1%", "f1_score": "85.9%", ...}
}
```

**CSV Results** (tuning_results.csv):
```
24 rows × 5 columns (all hyperparameter combinations tested)
```

**Professional Report** (SCENE_CLASSIFICATION_PROGRESS_REPORT.md):
- Executive summary
- Model comparison analysis
- Hyperparameter tuning insights
- Deployment recommendations

**Visualizations**:
- model_comparison_visualization.png (4-panel chart)
- hyperparameter_tuning_visualization.png (4-panel chart)

---

## ⚡ Advanced Options

### Run Individual Stages

**Stage 1 Only - Baseline Comparison:**
```bash
python model_comparison.py
```
Output: `comparison_results/baseline_comparison.json`

**Stage 2 Only - Hyperparameter Tuning:**
```bash
python hyperparameter_tuning.py
```
Output: `hyperparameter_tuning/tuning_results.csv`

**Stage 3 Only - Report Generation:**
```bash
python generate_progress_report.py
```
Output: `progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md`

---

## ⚠️ Requirements Checklist

Before running, ensure:

- [ ] Python 3.8+ installed
- [ ] PyTorch installed: `pip install torch torchvision`
- [ ] Dependencies: `pip install scikit-learn pandas matplotlib seaborn`
- [ ] Dataset exists at: `modalities/context/data/scene/train` and `/val`
- [ ] GPU available (CPU works but 10-50× slower)
- [ ] ~5GB free disk space for outputs
- [ ] ~2 hours free time (or can interrupt/resume)

**Check GPU:**
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 🔧 Customization

### Modify Training Configuration

Edit `hyperparameter_tuning.py` around line 60:

```python
learning_rates = [0.001, 0.0005, 0.0001, 0.00005]  # Change these
batch_sizes = [16, 32, 64]                          # Or these
optimizers = ['adam', 'sgd']                        # Or these
```

### Modify Number of Epochs

```python
EPOCHS = 15  # Change this value (line ~30)
```

### Use Smaller Training Set (Faster testing)

Comment out data loading and use subset:
```python
# Around line 130 in model_comparison.py:
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=...,
#     sampler=torch.utils.data.RandomSampler(
#         train_dataset,
#         num_samples=500  # Use only 500 samples instead of all
#     )
# )
```

---

## 📋 Expected Timeline

| Stage | Time | GPU | What's Happening |
|-------|------|-----|-----------------|
| Stage 1 | ~45 min | Yes | Training 3 models × 15 epochs each |
| Stage 2 | ~90 min | Yes | Testing 24 hyperparameter combinations |
| Stage 3 | ~5 min | No | Analyzing results, generating report |
| **Total** | **~2.5 hrs** | - | - |

---

## ✅ Verification

After completion, verify files exist:

```bash
# Stage 1 Results
ls comparison_results/baseline_comparison.json

# Stage 2 Results
ls hyperparameter_tuning/tuning_results.csv
ls hyperparameter_tuning/best_config.json

# Stage 3 Results
ls progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md
ls progress_report/model_comparison_visualization.png
ls progress_report/hyperparameter_tuning_visualization.png
```

All should exist with file size > 0.

---

## 🎯 What's Next

1. ✓ Run the analysis (you are here)
2. Review the generated report
3. Check visualizations
4. Apply best_config to train final model
5. Deploy to robot

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Verify path: `modalities/context/data/scene/` exists |
| "GPU out of memory" | Reduce batch_size in config (e.g., 16 instead of 32) |
| "Takes too long" | Normal! Use `nohup` to run in background |
| Import errors | Run: `pip install -r requirements.txt` |

For more help, see [SCENE_ANALYSIS_GUIDE.md](SCENE_ANALYSIS_GUIDE.md).

---

## 🚀 Ready?

```bash
cd modalities/context/scene_classification/
python run_complete_analysis.py
```

Grab ☕ and come back in 2-4 hours!

---

**Next**: [Full Documentation](START_HERE.md) | [Technical Guide](SCENE_ANALYSIS_GUIDE.md)
