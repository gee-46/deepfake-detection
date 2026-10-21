"""
Training Pipeline for Deepfake Image Detection
Trains an EfficientNetB0-based binary classifier with transfer learning,
data augmentation, class imbalance compensation, early stopping, and fine-tuning.
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
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from src.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    TARGET_SIZE,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    RANDOM_SEED,
)
from src.preprocessing import load_dataset_splits
from src.model import build_deepfake_detector, compile_deepfake_model, unfreeze_for_finetuning

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def calculate_class_weights(train_dataset) -> dict:
    """
    Computes balanced class weights from the training dataset
    to handle any imbalance between Real (1) and Fake (0) samples.
    """
    logger.info("Computing class distributions for class weight balancing...")
    all_labels = []
    # Collect labels from batches
    for _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten().astype(int))

    labels_array = np.array(all_labels)
    unique_classes = np.unique(labels_array)

    if len(unique_classes) < 2:
        logger.warning("Only one class found in dataset! Defaulting to uniform weights.")
        return {0: 1.0, 1: 1.0}

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels_array,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    logger.info(f"Class distributions: Fake (0)={np.sum(labels_array == 0)}, Real (1)={np.sum(labels_array == 1)}")
    logger.info(f"Computed Class Weights: {class_weight_dict}")
    return class_weight_dict


def plot_and_save_training_curves(history_dict: dict, output_path: Path):
    """Generates and saves visual loss, accuracy, and AUC curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Deepfake Classifier Training History (EfficientNetB0)", fontsize=14, fontweight="bold")

    # 1. Loss
    axes[0].plot(epochs_range, history_dict["loss"], "b-o", label="Train Loss", linewidth=2)
    if "val_loss" in history_dict:
        axes[0].plot(epochs_range, history_dict["val_loss"], "r--s", label="Val Loss", linewidth=2)
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Accuracy
    axes[1].plot(epochs_range, history_dict["accuracy"], "b-o", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history_dict:
        axes[1].plot(epochs_range, history_dict["val_accuracy"], "r--s", label="Val Accuracy", linewidth=2)
    axes[1].set_title("Binary Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3. AUC
    if "auc" in history_dict:
        axes[2].plot(epochs_range, history_dict["auc"], "b-o", label="Train ROC-AUC", linewidth=2)
        if "val_auc" in history_dict:
            axes[2].plot(epochs_range, history_dict["val_auc"], "r--s", label="Val ROC-AUC", linewidth=2)
        axes[2].set_title("Area Under ROC Curve (AUC)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC")
        axes[2].set_ylim([0.0, 1.05])
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    logger.info(f"Training history curves saved to {output_path}")


def train(
    dataset_dir: Path = DATASET_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
):
    """Executes the complete transfer learning and optional fine-tuning training pipeline."""
    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset splits from {dataset_dir} (Batch Size={batch_size}, Size={TARGET_SIZE})")
    try:
        train_ds, val_ds, _ = load_dataset_splits(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            img_size=TARGET_SIZE,
            seed=RANDOM_SEED,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please run `python prepare_dataset.py --generate_sample_data` or provide a valid dataset.")
        sys.exit(1)

    # Class Weights for imbalance handling
    class_weights = calculate_class_weights(train_ds)

    # Step 1: Build & Compile Model with frozen backbone
    logger.info("Initializing EfficientNetB0 with ImageNet pretrained transfer learning...")
    model = build_deepfake_detector(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_augmentation=True)
    model = compile_deepfake_model(model, learning_rate=learning_rate)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
            mode="min",
        ),
    ]

    # Stage 1: Feature Extraction (Frozen Base)
    logger.info(f"--- Starting Stage 1 Training: Feature Extraction ({epochs} epochs) ---")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = {k: list(v) for k, v in history_stage1.history.items()}

    # Stage 2: Fine-Tuning (Optional unfreezing)
    if fine_tune:
        logger.info(f"--- Starting Stage 2 Training: Fine-Tuning ({fine_tune_epochs} epochs) ---")
        model = unfreeze_for_finetuning(model, unfreeze_layers=25, fine_tune_lr=fine_tune_lr)
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        for k, v in history_stage2.history.items():
            if k in combined_history:
                combined_history[k].extend(v)
            else:
                combined_history[k] = list(v)

    # Save Final Model
    logger.info(f"Saving final model to {FINAL_MODEL_PATH}...")
    model.save(str(FINAL_MODEL_PATH))

    # Also ensure best_model.keras exists (in case checkpoint wasn't triggered)
    if not BEST_MODEL_PATH.exists():
        logger.info(f"Saving best model to {BEST_MODEL_PATH}...")
        model.save(str(BEST_MODEL_PATH))

    # Save Training History and Graphs
    curves_path = REPORTS_DIR / "training_curves.png"
    plot_and_save_training_curves(combined_history, curves_path)

    metrics_path = REPORTS_DIR / "train_metrics.json"
    summary_metrics = {
        "final_train_loss": float(combined_history["loss"][-1]),
        "final_train_accuracy": float(combined_history["accuracy"][-1]),
        "final_val_loss": float(combined_history.get("val_loss", [0])[-1]),
        "final_val_accuracy": float(combined_history.get("val_accuracy", [0])[-1]),
        "epochs_trained": len(combined_history["loss"]),
        "best_model_path": str(BEST_MODEL_PATH),
        "final_model_path": str(FINAL_MODEL_PATH),
    }
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    logger.info(f"Saved training metrics summary to {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best Model Saved:  {BEST_MODEL_PATH}")
    print(f"Final Model Saved: {FINAL_MODEL_PATH}")
    print(f"Training Curves:   {curves_path}")
    print(f"Metrics Summary:   {metrics_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deepfake Image Classifier Training Pipeline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Perform fine-tuning stage")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--fine_tune_lr", type=float, default=FINE_TUNE_LEARNING_RATE, help="Fine-tuning learning rate")
    parser.add_argument("--data_dir", type=str, default=str(DATASET_DIR), help="Dataset root directory")

    args = parser.parse_args()

    train(
        dataset_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
    )


if __name__ == "__main__":
    main()
