"""
Automated Unit Tests for Inference Engine and Prediction Logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import DeepfakePredictor
from src.config import PREDICTION_DISCLAIMER


def test_missing_model_behavior(tmp_path):
    """Verify that predictor handles a missing model file gracefully without crashing."""
    non_existent = tmp_path / "does_not_exist.keras"
    predictor = DeepfakePredictor(model_path=non_existent)

    assert predictor.is_ready is False
    assert predictor.load_error is not None

    # Prediction should return clean structured error dict
    dummy_img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    result = predictor.predict(dummy_img)

    assert result["status"] == "error"
    assert "error" in result
    assert result["prediction"] == "UNKNOWN"
    assert result["disclaimer"] == PREDICTION_DISCLAIMER


def test_threshold_logic_fake_prediction():
    """Verify inference classifies as LIKELY DEEPFAKE when fake score exceeds threshold."""
    predictor = DeepfakePredictor.__new__(DeepfakePredictor)
    predictor.model_path = Path("mock_model.keras")
    predictor._load_error = None

    # Mock Keras model output: real probability = 0.20 -> fake score = 0.80
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.20]], dtype=np.float32)
    predictor.model = mock_model

    dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    result = predictor.predict(dummy_img, threshold=0.50)

    assert result["status"] == "success"
    assert result["prediction"] == "LIKELY DEEPFAKE"
    assert result["dominant_class"] == "fake"
    assert result["fake_score"] == 0.80
    assert result["real_score"] == 0.20
    assert result["confidence_percent"] == 80.0
    assert result["threshold"] == 0.50


def test_threshold_logic_real_prediction():
    """Verify inference classifies as LIKELY REAL when fake score is below threshold."""
    predictor = DeepfakePredictor.__new__(DeepfakePredictor)
    predictor.model_path = Path("mock_model.keras")
    predictor._load_error = None

    # Mock Keras model output: real probability = 0.90 -> fake score = 0.10
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.90]], dtype=np.float32)
    predictor.model = mock_model

    dummy_img = Image.new("RGB", (224, 224), color=(150, 200, 100))
    result = predictor.predict(dummy_img, threshold=0.50)

    assert result["status"] == "success"
    assert result["prediction"] == "LIKELY REAL"
    assert result["dominant_class"] == "real"
    assert result["fake_score"] == 0.10
    assert result["real_score"] == 0.90
    assert result["confidence_percent"] == 90.0


def test_custom_threshold_boundary():
    """Verify custom threshold sensitivity behaves accurately."""
    predictor = DeepfakePredictor.__new__(DeepfakePredictor)
    predictor.model_path = Path("mock_model.keras")
    predictor._load_error = None

    # Real prob = 0.60 -> Fake score = 0.40
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.60]], dtype=np.float32)
    predictor.model = mock_model

    dummy_img = Image.new("RGB", (224, 224))

    # At standard threshold 0.50, fake_score 0.40 is below threshold -> REAL
    res1 = predictor.predict(dummy_img, threshold=0.50)
    assert res1["prediction"] == "LIKELY REAL"

    # At strict threshold 0.35, fake_score 0.40 exceeds threshold -> DEEPFAKE
    res2 = predictor.predict(dummy_img, threshold=0.35)
    assert res2["prediction"] == "LIKELY DEEPFAKE"
