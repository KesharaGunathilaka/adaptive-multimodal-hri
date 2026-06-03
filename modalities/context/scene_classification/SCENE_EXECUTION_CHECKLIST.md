## Scene Classification - Execution Checklist

**Complete before, during, and after running the analysis framework**

---

## ✅ PRE-EXECUTION CHECKLIST

### Environment Setup
- [ ] Python 3.8+ installed
  ```bash
  python --version  # Should be 3.8 or higher
  ```
- [ ] PyTorch installed with GPU support
  ```bash
  python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available())"
  ```
- [ ] Required packages installed
  ```bash
  pip install scikit-learn pandas matplotlib seaborn
  ```

### Dataset Verification
- [ ] Dataset path exists: `modalities/context/data/scene/`
- [ ] Train subdirectories present:
  - [ ] `train/classroom/` (images present)
  - [ ] `train/kitchen/` (images present)
  - [ ] `train/office/` (images present)
- [ ] Validation subdirectories present:
  - [ ] `val/classroom/` (images present)
  - [ ] `val/kitchen/` (images present)
  - [ ] `val/office/` (images present)
- [ ] All images are readable (not corrupted)
  ```bash
  # Quick check: count images in each directory
  ls modalities/context/data/scene/train/classroom/ | wc -l  # Should be > 0
  ```

### System Resources
- [ ] Sufficient disk space (~5GB free)
  ```bash
  df -h  # Check available space
  ```
- [ ] Sufficient RAM (8GB minimum, 16GB recommended)
  ```bash
  # Windows: Check System Information
  # Linux: free -h
  # Mac: vm_stat
  ```
- [ ] GPU available (recommended, CPU works but slow)
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] No other GPU-intensive processes running
  ```bash
  # Windows: Task Manager → GPU
  # Linux: nvidia-smi
  ```

### Code Files Present
- [ ] `model_comparison.py` exists
- [ ] `hyperparameter_tuning.py` exists
- [ ] `generate_progress_report.py` exists
- [ ] `run_complete_analysis.py` exists
- [ ] `scene_model.py` exists (model definition)
- [ ] `config.py` exists (configuration)

### Working Directory
- [ ] Located in: `modalities/context/scene_classification/`
  ```bash
  pwd  # Should show scene_classification directory
  ```
- [ ] Can write to this directory (permissions check)
  ```bash
  touch test_write.txt && rm test_write.txt  # Should succeed
  ```

---

## 🚀 EXECUTION CHECKLIST

### Starting the Analysis

**Phase 1: Launch**
- [ ] Terminal open and in correct directory
- [ ] All environment variables set (if needed)
- [ ] Run master script:
  ```bash
  python run_complete_analysis.py
  ```
- [ ] Script starts and asks for confirmation
- [ ] Type "yes" to proceed

**Phase 2: Stage 1 - Baseline Comparison**
- [ ] Script shows: "STAGE 1: BASELINE MODEL COMPARISON"
- [ ] Training begins with progress bars
- [ ] Expected time: ~45 minutes
- [ ] Monitor output:
  - [ ] MobileNetV3Small: Training starts
  - [ ] ResNet18: Training starts
  - [ ] EfficientNet-B0: Training starts
  - [ ] Epoch progress shown (1/15, 2/15, ...)
  - [ ] Validation metrics displayed
  - [ ] Final accuracies shown

**Phase 3: Stage 2 - Hyperparameter Tuning**
- [ ] Script shows: "STAGE 2: HYPERPARAMETER TUNING"
- [ ] Grid search begins (24 total trials)
- [ ] Expected time: ~90 minutes
- [ ] Monitor output:
  - [ ] Trial counter visible (Trial 1/24, 2/24, ...)
  - [ ] Hyperparameters shown: LR, BS, Opt
  - [ ] Progress bars for each trial
  - [ ] Best accuracy so far updated
  - [ ] All 24 trials complete

**Phase 4: Stage 3 - Report Generation**
- [ ] Script shows: "STAGE 3: PROGRESS REPORT GENERATION"
- [ ] Results being processed
- [ ] Expected time: ~5 minutes
- [ ] Monitor output:
  - [ ] "Loading results..."
  - [ ] "Generating main report..."
  - [ ] "Generating visualizations..."
  - [ ] PNG files saved confirmation
  - [ ] "Complete!" message

### During Execution

**Monitoring**
- [ ] Output scrolling normally (not hung)
- [ ] GPU utilization good (if using GPU)
  ```bash
  # In separate terminal on Linux:
  watch -n 1 nvidia-smi
  ```
- [ ] No error messages appearing
- [ ] No "Out of Memory" errors
- [ ] Temperature reasonable (GPU < 85°C)

**Disk Usage**
- [ ] Output directories created:
  - [ ] `comparison_results/` appearing
  - [ ] `hyperparameter_tuning/` appearing
  - [ ] `progress_report/` appearing
- [ ] File sizes growing:
  - [ ] CSV file getting larger (adding rows)
  - [ ] PNG files creating

**Time Tracking**
- [ ] Note start time: `_______`
- [ ] Check elapsed time at each stage:
  - [ ] Stage 1 elapsed: ~45 min
  - [ ] Stage 2 elapsed: ~90 min  
  - [ ] Stage 3 elapsed: ~5 min
- [ ] Total time reasonable (~2.5 hours)

### Handling Issues During Execution

**If Process Stops or Errors**

1. **Out of Memory Error**
   - [ ] Note the error message
   - [ ] Reduce batch size in `hyperparameter_tuning.py` (line ~60)
   - [ ] Change: `batch_sizes = [16, 32]` (remove 64)
   - [ ] Restart from that stage

2. **Dataset Not Found Error**
   - [ ] Check path: `modalities/context/data/scene/`
   - [ ] Verify subdirectories exist
   - [ ] List files to confirm:
     ```bash
     ls modalities/context/data/scene/train/classroom/
     ```
   - [ ] May need to create symbolic link if data is elsewhere

3. **GPU Out of Memory**
   - [ ] Reduce BATCH_SIZE in scripts (32 → 16)
   - [ ] Run on CPU instead (slower but works):
     - Edit scripts, change: `device = torch.device("cpu")`

4. **Interrupted Execution**
   - [ ] Manually run individual stages
   - [ ] Don't re-run completed stages
   - [ ] Results preserved in output directories

---

## ✅ POST-EXECUTION CHECKLIST

### Output Verification

**Stage 1 Results**
- [ ] File exists: `comparison_results/baseline_comparison.json`
- [ ] File size > 1 KB (not empty)
- [ ] Contains 3 models: MobileNetV3Small, ResNet18, EfficientNet-B0
- [ ] Each has metrics: accuracy, f1_score, inference_mean_ms
- [ ] Accuracy values between 70-95%

**Stage 2 Results**
- [ ] File exists: `hyperparameter_tuning/tuning_results.csv`
- [ ] File has 24 rows (all combinations tested)
- [ ] Columns include: learning_rate, batch_size, optimizer, accuracy, f1_score
- [ ] File exists: `hyperparameter_tuning/best_config.json`
- [ ] Best config specifies: learning_rate, batch_size, optimizer

**Stage 3 Results**
- [ ] File exists: `progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md`
- [ ] File size > 20 KB (substantial report)
- [ ] Contains sections:
  - [ ] Executive Summary
  - [ ] Baseline Model Comparison
  - [ ] Hyperparameter Tuning Analysis
  - [ ] Technical Specifications
  - [ ] Deployment Considerations
  - [ ] Recommendations
- [ ] File exists: `progress_report/model_comparison_visualization.png`
- [ ] File size > 100 KB (proper image)
- [ ] File exists: `progress_report/hyperparameter_tuning_visualization.png`
- [ ] File size > 100 KB (proper image)

### Results Review

**Report Content**
- [ ] Open report in text editor:
  ```bash
  cat progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md
  ```
- [ ] Verify content:
  - [ ] Clear executive summary
  - [ ] Model comparison table readable
  - [ ] Best configuration identified
  - [ ] Metrics for each model shown
  - [ ] Recommendations present

**Visualization Quality**
- [ ] Open images in image viewer:
  ```bash
  # Windows: start progress_report/model_comparison_visualization.png
  # Linux: display progress_report/model_comparison_visualization.png
  # Mac: open progress_report/model_comparison_visualization.png
  ```
- [ ] Verify visualizations:
  - [ ] 4 charts visible in model comparison
  - [ ] 4 charts visible in tuning analysis
  - [ ] Legends clear and readable
  - [ ] Axes labeled
  - [ ] Colors distinguishable

**Results Sanity Check**
- [ ] Accuracy values realistic:
  - [ ] All > 70% (reasonable)
  - [ ] All < 100% (not overfit)
- [ ] Inference times reasonable:
  - [ ] < 100ms (fast enough for real-time)
  - [ ] Different models have different speeds
- [ ] Best config identified clearly
- [ ] Hyperparameter ranges explored

### Performance Metrics Summary

Create summary in notes:
- [ ] Best Model: `_________________`
- [ ] Best Accuracy: `_________________`
- [ ] Best Learning Rate: `_________________`
- [ ] Best Batch Size: `_________________`
- [ ] Best Optimizer: `_________________`
- [ ] Inference Time: `_________________` ms
- [ ] Model Size: `_________________` MB

### Cleanup (Optional)

- [ ] Delete intermediate files (keep results)
- [ ] Archive output directory if needed:
  ```bash
  tar -czf scene_analysis_results.tar.gz comparison_results/ hyperparameter_tuning/ progress_report/
  ```
- [ ] Back up results to external storage
- [ ] Update project documentation with findings

---

## 🎯 NEXT STEPS

### Using Results

**1. Review the Generated Report**
```bash
cat progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md
```
- [ ] Read entire report
- [ ] Understand model selection rationale
- [ ] Note deployment considerations

**2. Train Final Model**
- [ ] Use best_config hyperparameters from tuning
- [ ] Train on full dataset (train + val)
- [ ] Save best checkpoint
- [ ] Document configuration

**3. Evaluate Performance**
- [ ] Test on hold-out test set (if available)
- [ ] Measure edge cases:
  - [ ] Low lighting
  - [ ] Occlusions
  - [ ] Different camera angles
- [ ] Compare to expected performance

**4. Prepare for Deployment**
- [ ] Export model to deployment format
- [ ] Create inference wrapper
- [ ] Integrate with emotion/gesture models
- [ ] Plan real-time processing loop

### Integration Points

**Emotion Recognition**:
- [ ] Scene context improves emotion interpretation
- [ ] Different emotion baselines per scene
- [ ] Update emotion model with scene features

**Robot Behavior**:
- [ ] Adapt interaction strategy per scene
- [ ] Different greeting gestures per environment
- [ ] Scene-aware response generation

---

## 📞 TROUBLESHOOTING REFERENCE

| Issue | Check | Fix |
|-------|-------|-----|
| "Dataset not found" | Path exists | Create symbolic link |
| GPU out of memory | VRAM usage | Reduce batch_size |
| Slow execution | GPU active? | Check nvidia-smi |
| Missing files | Output directories | Verify permissions |
| Low accuracy | Data quality | Check image files |
| Corrupted images | File format | Verify image validity |

---

## 📋 SIGN-OFF

- [ ] All pre-execution checks complete
- [ ] Execution monitoring log created
- [ ] Post-execution verification done
- [ ] Results reviewed and validated
- [ ] Next steps planned
- [ ] Documentation updated

**Analysis Date**: `_______`
**Total Time**: `_______` hours
**Notes**: 
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

**Previous**: [Quick Start](SCENE_QUICK_START.md)  
**Next**: [Technical Analysis Guide](SCENE_ANALYSIS_GUIDE.md)
