"""
MASTER EXECUTION SCRIPT - Scene Classification Analysis
Runs complete analysis pipeline in sequence:
1. Baseline Model Comparison
2. Hyperparameter Tuning
3. Progress Report Generation
"""

import os
import subprocess
import sys
import time
from datetime import datetime


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def run_script(script_name, description):
    """Run a Python script and track execution"""
    print_header(description)
    print(f"Starting: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name], check=True, capture_output=False, text=True
        )
        print(f"\n✓ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {description} - ERROR")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print_header("SCENE CLASSIFICATION MODEL - COMPLETE ANALYSIS PIPELINE")

    print("""
This script will execute the complete scene classification analysis:

STAGE 1: Baseline Comparison
  - Compares MobileNetV3Small, ResNet18, and EfficientNet-B0
  - Tests with identical hyperparameters
  - Generates model statistics and performance metrics

STAGE 2: Hyperparameter Tuning
  - Grid search over 24 hyperparameter combinations
  - Tests learning rates, batch sizes, and optimizers
  - Identifies optimal configuration for MobileNetV3Small

STAGE 3: Progress Report Generation
  - Compiles results from stages 1 & 2
  - Generates comprehensive markdown report
  - Creates comparison visualizations
  - Produces deployment recommendations

TOTAL ESTIMATED TIME: 2-4 hours (depends on your GPU)

""")

    input_msg = "Ready to start? (yes/no): "
    response = input(input_msg).lower().strip()

    if response not in ["yes", "y"]:
        print("Cancelled.")
        sys.exit(0)

    start_time = time.time()
    all_success = True

    # Stage 1: Baseline Comparison
    success1 = run_script("model_comparison.py", "STAGE 1: BASELINE MODEL COMPARISON")
    all_success = all_success and success1

    if success1:
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed / 60:.1f} minutes")

    # Stage 2: Hyperparameter Tuning
    if all_success:
        success2 = run_script(
            "hyperparameter_tuning.py", "STAGE 2: HYPERPARAMETER TUNING"
        )
        all_success = all_success and success2

        if success2:
            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed / 60:.1f} minutes")

    # Stage 3: Progress Report
    if all_success:
        success3 = run_script(
            "generate_progress_report.py", "STAGE 3: PROGRESS REPORT GENERATION"
        )
        all_success = all_success and success3

    # Final Summary
    total_elapsed = time.time() - start_time

    print_header("PIPELINE EXECUTION SUMMARY")

    print(
        f"Total Execution Time: {total_elapsed / 60:.1f} minutes ({total_elapsed / 3600:.1f} hours)"
    )
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if all_success:
        print("✓ ALL STAGES COMPLETED SUCCESSFULLY\n")
        print("Output files generated in:")
        print("  - comparison_results/baseline_comparison.json")
        print("  - hyperparameter_tuning/tuning_results.csv")
        print("  - hyperparameter_tuning/best_config.json")
        print("  - progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md")
        print("  - progress_report/model_comparison_visualization.png")
        print("  - progress_report/hyperparameter_tuning_visualization.png")
        print("\nRecommendation:")
        print("  1. Review: progress_report/SCENE_CLASSIFICATION_PROGRESS_REPORT.md")
        print("  2. Check visualizations in progress_report/ directory")
        print(
            "  3. Use best_config from hyperparameter_tuning/ for final model training"
        )
    else:
        print("✗ SOME STAGES FAILED")
        print("Please review error messages above.")
        sys.exit(1)
