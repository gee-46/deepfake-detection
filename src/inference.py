"""
Inference Pipeline for Deepfake Image Detection
Encapsulates model loading, validation, score computation, and structured prediction reporting.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import io
import numpy as np
from PIL import Image

from src.config import (
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    DEFAULT_THRESHOLD,
    CLASS_NAMES,
    LABEL_FAKE,
    LABEL_REAL,
    PREDICTION_DISCLAIMER,
)
from src.preprocessing import validate_image_file, preprocess_image_for_inference


class DeepfakePredictor:
    """
    Inference manager for deepfake detection model.
    Loads and caches the trained Keras model, evaluates images, and produces structured reports.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self._load_error: Optional[str] = None
        self._attempt_load()

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolves model path: honors explicit path if specified, otherwise checks default model locations."""
        if explicit_path is not None:
            return Path(explicit_path)

        if BEST_MODEL_PATH.exists():
            return BEST_MODEL_PATH
        if FINAL_MODEL_PATH.exists():
            return FINAL_MODEL_PATH

        return None

    def _attempt_load(self) -> bool:
        """Attempts to load Keras model into memory."""
        if not self.model_path or not self.model_path.exists():
            self._load_error = (
                f"Model file not found. Checked: {BEST_MODEL_PATH} and {FINAL_MODEL_PATH}. "
                "Please train a model using `python train.py` first."
            )
            return False

        try:
            import tensorflow as tf

            # Load trained model
            self.model = tf.keras.models.load_model(str(self.model_path))
            self._loaded_mtime = self.model_path.stat().st_mtime
            self._load_error = None
            return True
        except Exception as e:
            self._load_error = f"Failed to load model from {self.model_path}: {str(e)}"
            self.model = None
            return False

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and ready for inference."""
        return self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def reload(self) -> bool:
        """Manually forces a model reload from disk."""
        self.model_path = self._resolve_model_path(self.model_path)
        return self._attempt_load()

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Runs inference on a single image.
        
        Args:
            image_input: File path, raw bytes, BytesIO buffer, or PIL Image.
            threshold: Decision boundary for the fake-class score (default 0.50).
            
        Returns:
            Dictionary containing prediction label, scores, threshold, confidence, explanation, and metadata.
        """
        # Auto-reload if model on disk was retrained/updated
        if self.model_path and self.model_path.exists():
            current_mtime = self.model_path.stat().st_mtime
            if current_mtime > getattr(self, "_loaded_mtime", 0):
                self._attempt_load()

        # Ensure model is available
        if not self.is_ready:
            # Re-attempt load if a model was recently placed
            self.model_path = self._resolve_model_path(self.model_path)
            if not self._attempt_load():
                return {
                    "status": "error",
                    "error": self._load_error or "Model is not loaded.",
                    "prediction": "UNKNOWN",
                    "fake_score": 0.0,
                    "real_score": 0.0,
                    "threshold": threshold,
                    "disclaimer": PREDICTION_DISCLAIMER,
                }

        # Step 1: Preprocessing & Validation
        try:
            tensor_batch, metadata = preprocess_image_for_inference(image_input)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Image preprocessing failed: {str(e)}",
                "prediction": "INVALID_IMAGE",
                "fake_score": 0.0,
                "real_score": 0.0,
                "threshold": threshold,
                "disclaimer": PREDICTION_DISCLAIMER,
            }

        # Step 2: Forward Pass
        try:
            # Sigmoid output in [0.0, 1.0] corresponds to Real class (index 1)
            raw_prediction = self.model.predict(tensor_batch, verbose=0)
            real_score = float(raw_prediction[0][0])
            # Bound numerically within [0.0, 1.0]
            real_score = max(0.0, min(1.0, real_score))
            fake_score = 1.0 - real_score
        except Exception as e:
            return {
                "status": "error",
                "error": f"Inference execution failed: {str(e)}",
                "prediction": "INFERENCE_ERROR",
                "fake_score": 0.0,
                "real_score": 0.0,
                "threshold": threshold,
                "disclaimer": PREDICTION_DISCLAIMER,
            }

        # Step 3: Decision Boundary
        # Requirement: If fake-class score exceeds configured threshold -> "LIKELY DEEPFAKE", else "LIKELY REAL"
        if fake_score >= threshold:
            prediction_label = "LIKELY DEEPFAKE"
            dominant_class = "fake"
            confidence_score = fake_score
            margin = fake_score - threshold
            explanation = (
                f"The model detected manipulation artifacts or unnatural facial patterns. "
                f"The synthetic/fake score ({fake_score * 100:.2f}%) exceeds the decision threshold "
                f"of {threshold * 100:.1f}% by {margin * 100:.2f}%."
            )
        else:
            prediction_label = "LIKELY REAL"
            dominant_class = "real"
            confidence_score = real_score
            margin = threshold - fake_score
            explanation = (
                f"The model found authentic visual coherence and skin texture consistency. "
                f"The authenticity/real score ({real_score * 100:.2f}%) comfortably places this image "
                f"below the synthetic threshold ({threshold * 100:.1f}%)."
            )

        return {
            "status": "success",
            "prediction": prediction_label,
            "dominant_class": dominant_class,
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "fake_score_percent": round(fake_score * 100, 2),
            "real_score_percent": round(real_score * 100, 2),
            "confidence_percent": round(confidence_score * 100, 2),
            "threshold": threshold,
            "threshold_percent": round(threshold * 100, 1),
            "explanation": explanation,
            "metadata": metadata,
            "disclaimer": PREDICTION_DISCLAIMER,
            "model_path": str(self.model_path),
        }
