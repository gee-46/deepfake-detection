# Deepfake Image Detection Configuration

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Model Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

# Input Image Specifications
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Class Label Definitions (strictly consistent across train, eval, and inference)
# fake = 0, real = 1
CLASS_NAMES = ["fake", "real"]
LABEL_FAKE = 0
LABEL_REAL = 1
CLASS_MAPPING = {
    LABEL_FAKE: "fake",
    LABEL_REAL: "real",
}

# Training Hyperparameter Defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_DROPOUT_RATE = 0.3
RANDOM_SEED = 42

# Inference Defaults
DEFAULT_THRESHOLD = 0.50  # Threshold applied to Fake score for binary decision
MAX_UPLOAD_SIZE_MB = 15
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Disclaimer Text
PREDICTION_DISCLAIMER = (
    "This result is an automated model prediction, not definitive proof of image authenticity. "
    "The model may make mistakes on unseen manipulation techniques, AI-generated images, "
    "compressed images, and unfamiliar image sources."
)
