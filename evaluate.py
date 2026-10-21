"""
Evaluation Pipeline for Deepfake Image Detection
Evaluates the trained model on an independent test dataset and generates
accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curve.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from src.config import (
    DATASET_DIR,
    TEST_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    REPORTS_DIR,
    TARGET_SIZE,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plots and saves the confusion matrix heatmap using Seaborn."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Deepfake Detector Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, roc_auc: float, output_path: Path):
    """Plots and saves ROC curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    logger.info(f"ROC curve plot saved to {output_path}")


def evaluate_model(
    model_path: Path,
    test_dir: Path = TEST_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Evaluates the model on independent test dataset."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error(f"Target model not found at: {model_path}")
        sys.exit(1)

    if not test_dir.exists():
        logger.error(f"Test directory not found at: {test_dir}")
        sys.exit(1)

    logger.info(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    logger.info(f"Loading independent test dataset from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,  # fake=0, real=1
        color_mode="rgb",
        batch_size=batch_size,
        image_size=TARGET_SIZE,
        shuffle=False,
    )

    # Collect ground truth and model predictions
    y_true_list = []
    y_pred_prob_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy().flatten())
        y_pred_prob_list.extend(preds.flatten())

    y_true = np.array(y_true_list).astype(int)
    # y_pred_prob is P(real). Fake probability is (1 - P(real)).
    # If Fake probability >= threshold -> class 0 (Fake), else class 1 (Real)
    # Equivalently, Real predicted if P(real) > (1 - threshold)
    y_pred_real_prob = np.array(y_pred_prob_list)
    y_pred_fake_prob = 1.0 - y_pred_real_prob

    # Binary decision based on configured threshold:
    # fake = 0 if y_pred_fake_prob >= threshold else 1 (real)
    y_pred = np.where(y_pred_fake_prob >= threshold, 0, 1)

    # Metrics computation
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ROC AUC calculation
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_real_prob))
    except Exception as e:
        logger.warning(f"ROC-AUC could not be computed (single class in test set?): {e}")
        roc_auc = 0.5

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # Save Confusion Matrix Heatmap
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    # Save ROC Curve Plot
    roc_path = REPORTS_DIR / "roc_curve.png"
    plot_roc_curve(y_true, y_pred_real_prob, roc_auc, roc_path)

    # Save Text Report
    report_text_path = REPORTS_DIR / "evaluation_report.txt"
    with open(report_text_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DEEPFAKE IMAGE DETECTOR - TEST EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model Evaluated: {model_path.name}\n")
        f.write(f"Test Directory:  {test_dir}\n")
        f.write(f"Decision Threshold: {threshold:.2f} (Fake Score)\n")
        f.write(f"Total Test Samples: {len(y_true)}\n\n")
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  [TN (True Fake) = {cm[0, 0]} | FP (False Real) = {cm[0, 1]}]\n")
        f.write(f"  [FN (False Fake) = {cm[1, 0]} | TP (True Real) = {cm[1, 1]}]\n\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(clf_report + "\n")
        f.write("=" * 60 + "\n")
        f.write("IMPORTANT INTERPRETATION NOTES:\n")
        f.write(
            "Test results directly depend on dataset quality, distribution balance,\n"
            "and whether the test images faithfully represent in-the-wild manipulation\n"
            "methods (GANs, diffusion models, FaceForensics++ deepfakes).\n"
            "High test set accuracy does not imply 100% real-world generalization.\n"
        )
        f.write("=" * 60 + "\n")

    # Save JSON Metrics
    metrics_json_path = REPORTS_DIR / "evaluation_metrics.json"
    metrics_data = {
        "model_file": str(model_path),
        "test_samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "true_fakes": int(cm[0, 0]),
            "false_reals": int(cm[0, 1]),
            "false_fakes": int(cm[1, 0]),
            "true_reals": int(cm[1, 1]),
        },
        "threshold": float(threshold),
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Print summary to stdout
    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"Report Text:     {report_text_path}")
    print(f"Report JSON:     {metrics_json_path}")
    print(f"Confusion Matrix:{cm_path}")
    print(f"ROC Curve Plot:  {roc_path}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deepfake Image Classifier Evaluation Pipeline")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained .keras model file")
    parser.add_argument("--test_dir", type=str, default=str(TEST_DIR), help="Test dataset directory")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold for fake score")

    args = parser.parse_args()

    # Model path resolution: prioritize argument -> best_model.keras -> final_model.keras
    if args.model_path:
        model_p = Path(args.model_path)
    elif BEST_MODEL_PATH.exists():
        model_p = BEST_MODEL_PATH
    elif FINAL_MODEL_PATH.exists():
        model_p = FINAL_MODEL_PATH
    else:
        logger.error("No model file found. Please train a model first using `python train.py`.")
        sys.exit(1)

    evaluate_model(
        model_path=model_p,
        test_dir=Path(args.test_dir),
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
