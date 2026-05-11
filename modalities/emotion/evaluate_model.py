import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.mobilenet_emotion import get_model
from utils.transforms import get_test_transforms
from config import EMOTION_LABELS, NUM_CLASSES, BATCH_SIZE
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================================
    # 1. LOAD MODEL
    # ============================================================================
    model = get_model()
    model.load_state_dict(torch.load("checkpoints/model_v1.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded from checkpoints/model_v1.pth")

    # ============================================================================
    # 2. LOAD TEST DATA
    # ============================================================================
    test_dataset = ImageFolder("data/test", transform=get_test_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    # ============================================================================
    # 3. GENERATE PREDICTIONS
    # ============================================================================
    all_preds = []
    all_labels = []
    all_confidences = []
    misclassified = []

    print("\nGenerating predictions...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Track misclassifications
            for i in range(len(predicted)):
                if predicted[i] != labels[i]:
                    misclassified.append(
                        {
                            "true_label": EMOTION_LABELS[labels[i]],
                            "predicted_label": EMOTION_LABELS[predicted[i]],
                            "confidence": confidences[i].item(),
                            "correct": False,
                        }
                    )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # ============================================================================
    # 4. OVERALL METRICS
    # ============================================================================
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average="weighted")
    overall_precision = precision_score(all_labels, all_preds, average="weighted")
    overall_recall = recall_score(all_labels, all_preds, average="weighted")

    print(f"\nAccuracy:  {overall_acc * 100:.2f}%")
    print(f"Precision: {overall_precision * 100:.2f}%")
    print(f"Recall:    {overall_recall * 100:.2f}%")
    print(f"F1-Score:  {overall_f1 * 100:.2f}%")

    # ============================================================================
    # 5. PER-CLASS METRICS
    # ============================================================================
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    class_report = classification_report(
        all_labels, all_preds, target_names=EMOTION_LABELS, digits=4, output_dict=True
    )

    print(
        "\n{:<15} {:<12} {:<12} {:<12} {:<12}".format(
            "Emotion", "Precision", "Recall", "F1-Score", "Support"
        )
    )
    print("-" * 70)

    per_class_data = []
for emotion_name in EMOTION_LABELS:
    # Use the emotion name string as the key instead of the index
    metrics = class_report[emotion_name]

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1-score"]
    support = int(metrics["support"])

    print(
        "{:<15} {:<12.4f} {:<12.4f} {:<12.4f} {:<12}".format(
            emotion_name, precision, recall, f1, support
        )
    )

    per_class_data.append(
        {
            "Emotion": emotion_name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )

    # ============================================================================
    # 6. CONFUSION MATRIX
    # ============================================================================
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # ============================================================================
    # 7. CONFIDENCE ANALYSIS
    # ============================================================================
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    correct_mask = all_preds == all_labels
    correct_confidences = all_confidences[correct_mask]
    incorrect_confidences = all_confidences[~correct_mask]

    print(f"\nTotal Predictions: {len(all_preds)}")
    print(
        f"Correct: {len(correct_confidences)} ({len(correct_confidences) / len(all_preds) * 100:.2f}%)"
    )
    print(
        f"Incorrect: {len(incorrect_confidences)} ({len(incorrect_confidences) / len(all_preds) * 100:.2f}%)"
    )

    print(f"\nCorrect Predictions:")
    print(f"  Mean Confidence: {correct_confidences.mean():.4f}")
    print(f"  Std Confidence:  {correct_confidences.std():.4f}")
    print(f"  Min Confidence:  {correct_confidences.min():.4f}")
    print(f"  Max Confidence:  {correct_confidences.max():.4f}")

    print(f"\nIncorrect Predictions:")
    if len(incorrect_confidences) > 0:
        print(f"  Mean Confidence: {incorrect_confidences.mean():.4f}")
        print(f"  Std Confidence:  {incorrect_confidences.std():.4f}")
        print(f"  Min Confidence:  {incorrect_confidences.min():.4f}")
        print(f"  Max Confidence:  {incorrect_confidences.max():.4f}")
    else:
        print(f"  No incorrect predictions!")

    # ============================================================================
    # 8. MISCLASSIFICATION ANALYSIS
    # ============================================================================
    if misclassified:
        print("\n" + "=" * 70)
        print("TOP 10 MISCLASSIFICATIONS (Highest Confidence Errors)")
        print("=" * 70)

        sorted_misclass = sorted(
            misclassified, key=lambda x: x["confidence"], reverse=True
        )[:10]
        print(
            f"\n{'{:<15} {:<15} {:<10}'.format('True Label', 'Predicted', 'Confidence')}"
        )
        print("-" * 40)
        for item in sorted_misclass:
            print(
                f"{item['true_label']:<15} {item['predicted_label']:<15} {item['confidence']:.4f}"
            )

    # ============================================================================
    # 9. SAVE RESULTS
    # ============================================================================
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Save per-class metrics
    per_class_df = pd.DataFrame(per_class_data)
    per_class_df.to_csv(f"{results_dir}/per_class_metrics.csv", index=False)
    print(f"\n✓ Saved: {results_dir}/per_class_metrics.csv")

    # Save overall metrics
    overall_metrics = {
        "Accuracy": overall_acc,
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-Score": overall_f1,
        "Total_Samples": len(all_preds),
        "Correct": len(correct_confidences),
        "Incorrect": len(incorrect_confidences),
    }
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(f"{results_dir}/overall_metrics.csv", index=False)
    print(f"✓ Saved: {results_dir}/overall_metrics.csv")

    # Save misclassifications
    if misclassified:
        misclass_df = pd.DataFrame(misclassified)
        misclass_df.to_csv(f"{results_dir}/misclassifications.csv", index=False)
        print(f"✓ Saved: {results_dir}/misclassifications.csv")

    # ============================================================================
    # 10. GENERATE PLOTS
    # ============================================================================
    print("\nGenerating visualizations...")

    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=150)
    print(f"✓ Saved: {results_dir}/confusion_matrix.png")
    plt.close()

    # Plot 2: Per-Class F1 Scores
    plt.figure(figsize=(10, 6))
    emotions = [d["Emotion"] for d in per_class_data]
    f1_scores = [d["F1-Score"] for d in per_class_data]
    colors = [
        "green" if f1 > 0.85 else "orange" if f1 > 0.7 else "red" for f1 in f1_scores
    ]
    plt.bar(emotions, f1_scores, color=colors)
    plt.axhline(y=0.85, color="r", linestyle="--", label="Target: 0.85")
    plt.ylabel("F1-Score")
    plt.title("Per-Class F1-Scores")
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/per_class_f1_scores.png", dpi=150)
    print(f"✓ Saved: {results_dir}/per_class_f1_scores.png")
    plt.close()

    # Plot 3: Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        correct_confidences,
        bins=30,
        alpha=0.7,
        label=f"Correct (n={len(correct_confidences)})",
        color="green",
    )
    if len(incorrect_confidences) > 0:
        plt.hist(
            incorrect_confidences,
            bins=30,
            alpha=0.7,
            label=f"Incorrect (n={len(incorrect_confidences)})",
            color="red",
        )
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confidence_distribution.png", dpi=150)
    print(f"✓ Saved: {results_dir}/confidence_distribution.png")
    plt.close()

    # Plot 4: Per-Class Recall
    plt.figure(figsize=(10, 6))
    recalls = [d["Recall"] for d in per_class_data]
    colors = [
        "green" if recall > 0.85 else "orange" if recall > 0.7 else "red"
        for recall in recalls
    ]
    plt.bar(emotions, recalls, color=colors)
    plt.axhline(y=0.85, color="r", linestyle="--", label="Target: 0.85")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall (Sensitivity)")
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/per_class_recall.png", dpi=150)
    print(f"✓ Saved: {results_dir}/per_class_recall.png")
    plt.close()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {results_dir}/")
    print("\nFiles generated:")
    print(f"  - per_class_metrics.csv")
    print(f"  - overall_metrics.csv")
    if misclassified:
        print(f"  - misclassifications.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_f1_scores.png")
    print(f"  - per_class_recall.png")
    print(f"  - confidence_distribution.png")
