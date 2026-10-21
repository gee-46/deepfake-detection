"""
Automated Unit Tests for Preprocessing and Image Validation
"""

import io
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from src.preprocessing import validate_image_file, preprocess_image_for_inference


def create_dummy_image(size=(100, 100), color=(120, 150, 180), mode="RGB", fmt="JPEG") -> bytes:
    """Helper to create dummy in-memory image bytes."""
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_validate_valid_jpeg():
    """Verify standard JPEG image is successfully validated."""
    img_bytes = create_dummy_image(size=(200, 200), mode="RGB", fmt="JPEG")
    is_valid, msg, img = validate_image_file(img_bytes)
    assert is_valid is True
    assert img is not None
    assert img.mode == "RGB"
    assert img.size == (200, 200)


def test_validate_valid_png():
    """Verify PNG image is successfully validated."""
    img_bytes = create_dummy_image(size=(150, 150), mode="RGB", fmt="PNG")
    is_valid, msg, img = validate_image_file(img_bytes)
    assert is_valid is True
    assert img is not None
    assert img.mode == "RGB"


def test_validate_rgba_conversion():
    """Verify RGBA images are converted to RGB mode."""
    img_bytes = create_dummy_image(size=(120, 120), mode="RGBA", fmt="PNG")
    is_valid, msg, img = validate_image_file(img_bytes)
    assert is_valid is True
    assert img.mode == "RGB"


def test_validate_corrupt_data():
    """Verify corrupt random byte stream is rejected cleanly without crashing."""
    corrupt_bytes = b"NOT_AN_IMAGE_HEADER_RANDOM_GARBAGE_BYTES_12345"
    is_valid, msg, img = validate_image_file(corrupt_bytes)
    assert is_valid is False
    assert img is None
    assert "Invalid or corrupt image" in msg or "cannot identify" in msg.lower()


def test_validate_undersized_image():
    """Verify images smaller than 32x32 are rejected."""
    small_bytes = create_dummy_image(size=(16, 16), mode="RGB", fmt="JPEG")
    is_valid, msg, img = validate_image_file(small_bytes)
    assert is_valid is False
    assert "too small" in msg.lower()


def test_preprocess_tensor_shape_and_dtype():
    """Verify preprocessed tensor has exact shape (1, 224, 224, 3) and float32 dtype."""
    img_bytes = create_dummy_image(size=(300, 450), mode="RGB", fmt="JPEG")
    tensor_batch, meta = preprocess_image_for_inference(img_bytes)

    assert tensor_batch.shape == (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert tensor_batch.dtype == np.float32
    assert meta["original_width"] == 300
    assert meta["original_height"] == 450
    assert meta["target_size"] == TARGET_SIZE
