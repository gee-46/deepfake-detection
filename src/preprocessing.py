"""
Preprocessing and Dataset Pipeline Utilities
Ensures consistent image transformations across training, evaluation, and inference.
"""

import io
import os
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    TARGET_SIZE,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    CLASS_NAMES,
    LABEL_FAKE,
    LABEL_REAL,
    RANDOM_SEED,
)


def validate_image_file(
    image_input: Union[str, Path, bytes, io.BytesIO],
    max_size_mb: float = MAX_UPLOAD_SIZE_MB,
) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates an image from a path, raw bytes, or BytesIO buffer.
    
    Checks:
    1. Readability & corruption
    2. File format (JPEG, PNG)
    3. File size constraints
    4. Image dimensions
    
    Returns:
        (is_valid, message, PIL Image in RGB mode or None)
    """
    try:
        # Check size if bytes or file path
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"File does not exist: {path}", None
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed ({max_size_mb} MB).", None
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file extension '{path.suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", None
            img = Image.open(path)
        elif isinstance(image_input, bytes):
            size_mb = len(image_input) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.2f} MB) exceeds limit of {max_size_mb} MB.", None
            img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            # BytesIO or file-like object (e.g. Streamlit UploadedFile)
            content = image_input.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.2f} MB) exceeds limit of {max_size_mb} MB.", None
            img = Image.open(io.BytesIO(content))
        else:
            return False, "Unsupported image input type.", None

        # Verify integrity
        img.verify()

        # Reopen for actual processing (verify closes/corrupts the handle in PIL)
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        else:
            img = Image.open(io.BytesIO(content))

        # Check format
        fmt = (img.format or "").upper()
        if fmt not in {"JPEG", "JPG", "PNG"}:
            return False, f"Unsupported image format '{fmt}'. Only JPEG and PNG are allowed.", None

        # Validate minimum dimensions
        if img.width < 32 or img.height < 32:
            return False, f"Image dimensions too small ({img.width}x{img.height}). Minimum is 32x32.", None

        # Convert to RGB (handles RGBA, grayscale, palette, CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return True, "Image is valid.", img

    except (UnidentifiedImageError, OSError, ValueError) as err:
        return False, f"Invalid or corrupt image: {str(err)}", None
    except Exception as err:
        return False, f"Unexpected error while opening image: {str(err)}", None


def preprocess_image_for_inference(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standardizes an image for inference with EfficientNetB0.
    
    1. Validates or converts input to RGB PIL Image.
    2. Resizes to target_size (224, 224) using high-quality Lanczos resampling.
    3. Converts to float32 NumPy array with shape (1, 224, 224, 3).
    4. Applies architecture-specific preprocessing.
    
    Returns:
        (preprocessed_tensor_batch, metadata_dict)
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
        orig_width, orig_height = img.width, img.height
    else:
        is_valid, msg, img = validate_image_file(image_input)
        if not is_valid or img is None:
            raise ValueError(f"Image preprocessing failed: {msg}")
        orig_width, orig_height = img.width, img.height

    # High-quality resize
    resized_img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)

    # Validate shape
    if img_array.shape != (target_size[0], target_size[1], IMG_CHANNELS):
        raise ValueError(f"Unexpected image array shape: {img_array.shape}")

    # Expand batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # Note: Keras EfficientNet models include built-in Rescaling([0, 255] -> normalized)
    # when initialized with include_top=False if using keras.applications.efficientnet.
    # We pass standard float32 RGB values in [0, 255].
    metadata = {
        "original_width": orig_width,
        "original_height": orig_height,
        "processed_shape": batch_array.shape,
        "dtype": str(batch_array.dtype),
        "target_size": target_size,
    }

    return batch_array, metadata


def get_augmentation_model():
    """
    Builds a Keras data augmentation pipeline.
    Applies mild, realistic spatial variations to prevent overfitting.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=RANDOM_SEED, name="aug_horizontal_flip"),
            layers.RandomRotation(0.06, fill_mode="nearest", seed=RANDOM_SEED, name="aug_rotation"),
            layers.RandomZoom(0.06, fill_mode="nearest", seed=RANDOM_SEED, name="aug_zoom"),
            layers.RandomTranslation(0.04, 0.04, fill_mode="nearest", seed=RANDOM_SEED, name="aug_translation"),
        ],
        name="data_augmentation",
    )


def load_dataset_splits(
    dataset_dir: Union[str, Path],
    batch_size: int = 16,
    img_size: Tuple[int, int] = TARGET_SIZE,
    seed: int = RANDOM_SEED,
):
    """
    Loads train, validation, and test datasets from directory structure:
        dataset/train/real/
        dataset/train/fake/
        dataset/validation/real/
        dataset/validation/fake/
        dataset/test/real/
        dataset/test/fake/
        
    Guarantees strict class ordering:
        index 0 -> 'fake'
        index 1 -> 'real'
    """
    import tensorflow as tf

    base_path = Path(dataset_dir)
    train_path = base_path / "train"
    val_path = base_path / "validation"
    test_path = base_path / "test"

    for p, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} directory at: {p}")
        for c in CLASS_NAMES:
            c_dir = p / c
            if not c_dir.exists():
                raise FileNotFoundError(f"Missing class folder '{c}' inside {p}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_path),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,  # Ensures fake=0, real=1
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(val_path),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_path),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds
