"""
Model Architecture Definition for Deepfake Image Detection
Implements Transfer Learning with EfficientNetB0 as the core feature extractor.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

from src.config import (
    INPUT_SHAPE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT_RATE,
)
from src.preprocessing import get_augmentation_model


def build_deepfake_detector(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    pretrained_weights: str = "imagenet",
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    include_augmentation: bool = True,
) -> tf.keras.Model:
    """
    Builds the binary deepfake classifier using transfer learning with EfficientNetB0.
    
    Architecture:
    1. Input layer (224, 224, 3)
    2. Data Augmentation layers (active only during training)
    3. Pretrained EfficientNetB0 backbone (frozen initially)
    4. Global Average Pooling 2D
    5. Batch Normalization
    6. Dense FC (128 units, ReLU)
    7. Dropout (0.3)
    8. Dense Output (1 unit, Sigmoid) -> Output probability of REAL class [0.0 = Fake, 1.0 = Real]
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Optional augmentation pipeline
    if include_augmentation:
        aug_pipeline = get_augmentation_model()
        x = aug_pipeline(inputs)
    else:
        x = inputs

    # Load EfficientNetB0 base
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=x,
            pooling=None,
        )
    except Exception as e:
        # Fallback if offline or network blocks ImageNet weight download
        print(f"[WARNING] Could not load weights='{pretrained_weights}': {e}. Initializing randomly.")
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling=None,
        )

    # Freeze base model for initial transfer learning stage
    base_model.trainable = False

    # Classification Head
    feature_map = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(feature_map)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dense(64, activation="relu", name="dense_features_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_DeepfakeDetector")
    return model


def compile_deepfake_model(
    model: tf.keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Compiles the model with binary cross-entropy, Adam optimizer, and comprehensive metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def unfreeze_for_finetuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 20,
    fine_tune_lr: float = 1e-5,
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of the EfficientNet base model for fine-tuning with a lower learning rate.
    Batch Normalization layers remain frozen to preserve running statistics.
    """
    # Find base model layer (EfficientNetB0)
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        # Layers may be flattened inside model
        layers_to_consider = [l for l in model.layers if "dense" not in l.name and "head" not in l.name]
    else:
        layers_to_consider = base_model.layers

    # Set base model trainable
    if base_model:
        base_model.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    for layer in layers_to_consider[:-unfreeze_layers]:
        layer.trainable = False
    for layer in layers_to_consider[-unfreeze_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Keep BatchNorm frozen
        else:
            layer.trainable = True

    # Recompile with smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
