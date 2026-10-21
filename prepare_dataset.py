"""
Dataset Preparation and Organization Script
Organizes raw image collections into train, validation, and test splits with strict class separation.
Detects corrupt images, prevents data leakage across splits, and supports synthetic benchmark generation.
"""

import os
import sys
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Ensure src modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES,
    RANDOM_SEED,
    ALLOWED_EXTENSIONS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_dataset")


def setup_directory_structure(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Creates the target train, validation, and test folder trees."""
    paths = {}
    for split in ["train", "validation", "test"]:
        paths[split] = {}
        for c in CLASS_NAMES:
            folder = base_dir / split / c
            folder.mkdir(parents=True, exist_ok=True)
            paths[split][c] = folder
    return paths


def is_image_readable(image_path: Path) -> bool:
    """Checks whether an image file is readable and uncorrupted."""
    try:
        if image_path.stat().st_size == 0:
            return False
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception as e:
        logger.warning(f"Corrupt or unreadable image skipped: {image_path} ({e})")
        return False


def extract_group_id(filename: str) -> str:
    """
    Extracts group/identity ID to prevent data leakage across splits.
    For FaceForensics++ (e.g. '045_321_0001.png'), the group is '045_321'.
    For Kaggle datasets (e.g. 'id10_frame20.jpg'), the group is 'id10'.
    Defaults to base filename without suffix if no delimiter exists.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def partition_groups(
    groups: List[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Partitions unique groups into train, validation, and test sets."""
    unique_groups = sorted(list(set(groups)))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_total = len(unique_groups)
    n_test = max(1, int(n_total * test_ratio)) if n_total >= 3 else 0
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])

    if not train_groups and unique_groups:
        train_groups = set(unique_groups)

    return train_groups, val_groups, test_groups


def ingest_from_source(
    source_dir: Path,
    dest_dir: Path = DATASET_DIR,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
    copy_files: bool = True,
) -> Dict[str, int]:
    """
    Ingests images from an existing downloaded dataset folder.
    Expects source_dir to have 'real' and 'fake' subfolders, or scans recursively.
    """
    logger.info(f"Scanning source dataset directory: {source_dir}")
    target_dirs = setup_directory_structure(dest_dir)
    stats = {"train_real": 0, "train_fake": 0, "val_real": 0, "val_fake": 0, "test_real": 0, "test_fake": 0, "skipped": 0}

    for label in CLASS_NAMES:
        label_dir = source_dir / label
        if not label_dir.exists():
            # Try alternate names (e.g., 'deepfake', 'original', 'manipulated')
            candidates = [p for p in source_dir.iterdir() if p.is_dir() and label in p.name.lower()]
            if candidates:
                label_dir = candidates[0]
            else:
                logger.warning(f"Could not find subfolder for class '{label}' in {source_dir}")
                continue

        # Collect all valid image paths
        all_images = [
            p for p in label_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        logger.info(f"Found {len(all_images)} raw images for class '{label}'")

        # Filter unreadable files and map by group
        valid_files = []
        groups = []
        for img_path in all_images:
            if is_image_readable(img_path):
                valid_files.append(img_path)
                groups.append(extract_group_id(img_path.name))
            else:
                stats["skipped"] += 1

        # Partition groups to avoid data leakage
        train_g, val_g, test_g = partition_groups(groups, val_ratio, test_ratio, seed)

        for img_path in valid_files:
            group = extract_group_id(img_path.name)
            if group in test_g and test_ratio > 0:
                split = "test"
            elif group in val_g and val_ratio > 0:
                split = "validation"
            else:
                split = "train"

            target_file = target_dirs[split][label] / img_path.name
            # Avoid collision
            if target_file.exists():
                target_file = target_dirs[split][label] / f"{img_path.stem}_{random.randint(1000, 9999)}{img_path.suffix}"

            if copy_files:
                shutil.copy2(img_path, target_file)
            else:
                shutil.move(img_path, target_file)

            stats[f"{split}_{label}"] += 1

    logger.info("Dataset ingestion completed.")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    return stats


def generate_synthetic_benchmark(
    dest_dir: Path = DATASET_DIR,
    samples_per_class: int = 50,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """
    Generates a calibrated synthetic benchmark dataset for local verification and pipeline testing.
    - 'real' images simulate coherent facial portraits with natural skin gradients, natural eyes, and organic lighting.
    - 'fake' images incorporate characteristic deepfake artifacts: blending boundaries, color-channel misalignment,
      frequency warping, and blur anomalies common in DeepFaceLab / FaceForensics++.
      
    This enables immediate end-to-end testing of training, evaluation, and Streamlit inference
    without waiting for multi-gigabyte external downloads.
    """
    logger.info(f"Generating synthetic benchmark dataset ({samples_per_class} per class)...")
    target_dirs = setup_directory_structure(dest_dir)
    np.random.seed(seed)
    random.seed(seed)

    splits = {
        "train": int(samples_per_class * 0.70),
        "validation": int(samples_per_class * 0.15),
        "test": samples_per_class - int(samples_per_class * 0.70) - int(samples_per_class * 0.15),
    }

    stats = {f"{s}_{c}": 0 for s in splits for c in CLASS_NAMES}

    for split, count in splits.items():
        for i in range(count):
            # 1. Generate Realistic Face-like Canvas
            real_img = _create_synthetic_face(is_fake=False, seed=seed + i * 13)
            real_path = target_dirs[split]["real"] / f"real_{split}_{i:04d}.jpg"
            real_img.save(real_path, format="JPEG", quality=95)
            stats[f"{split}_real"] += 1

            # 2. Generate Manipulated/Artifact Face-like Canvas
            fake_img = _create_synthetic_face(is_fake=True, seed=seed + i * 17)
            fake_path = target_dirs[split]["fake"] / f"fake_{split}_{i:04d}.jpg"
            fake_img.save(fake_path, format="JPEG", quality=90)
            stats[f"{split}_fake"] += 1

    logger.info("Synthetic benchmark generation finished:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    return stats


def _create_synthetic_face(is_fake: bool, seed: int) -> Image.Image:
    """
    Draws a structured portrait canvas simulating photographic portraits.
    - REAL: Coherent lighting, consistent Gaussian sensor grain, matching eye highlights, organic skin gradient.
    - FAKE: DeepFaceLab/FaceForensics++ manipulation signatures including boundary blending seams,
      upscaling blur mismatch in facial mask, localized chromatic aberration, and pupil asymmetry.
    """
    rng = np.random.RandomState(seed)
    width, height = 256, 256

    # 1. Background (studio gradients, indoor warm tones, or cool slate)
    bg_types = [
        (40, 50, 70),    # Slate blue
        (75, 55, 45),    # Warm studio
        (35, 40, 45),    # Charcoal dark
        (80, 85, 90),    # Neutral gray
        (50, 65, 55),    # Soft sage
    ]
    bg_base = bg_types[rng.randint(0, len(bg_types))]
    base_arr = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        factor = y / height
        base_arr[y, :, 0] = bg_base[0] * (1.0 - factor * 0.35)
        base_arr[y, :, 1] = bg_base[1] * (1.0 - factor * 0.30)
        base_arr[y, :, 2] = bg_base[2] * (1.0 - factor * 0.25)

    img = Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 2. Hair (varied silhouettes)
    hair_colors = [
        (25, 20, 18),   # Dark brown / black
        (65, 40, 25),   # Auburn
        (130, 95, 55),  # Honey brown
        (180, 150, 90), # Blonde
        (45, 30, 22),   # Deep chestnut
    ]
    hair_c = hair_colors[rng.randint(0, len(hair_colors))]
    draw.ellipse([36, 18, 220, 220], fill=hair_c)

    # 3. Face Oval (Diverse natural skin tone palettes: Fitzpatrick types I - VI)
    skin_palettes = [
        (245, 215, 195),  # Type I / II: Fair rosy
        (230, 190, 160),  # Type III: Warm peach
        (205, 160, 125),  # Type IV: Olive / golden
        (165, 115, 80),   # Type V: Warm bronze
        (110, 70, 45),    # Type VI: Deep mahogany
    ]
    skin_base = skin_palettes[rng.randint(0, len(skin_palettes))]
    # Add random individual nuance
    skin_r = int(np.clip(skin_base[0] + rng.randint(-8, 8), 60, 255))
    skin_g = int(np.clip(skin_base[1] + rng.randint(-8, 8), 40, 240))
    skin_b = int(np.clip(skin_base[2] + rng.randint(-8, 8), 30, 220))

    face_x1 = rng.randint(46, 52)
    face_y1 = rng.randint(38, 44)
    face_x2 = rng.randint(204, 210)
    face_y2 = rng.randint(218, 224)
    face_box = [face_x1, face_y1, face_x2, face_y2]
    draw.ellipse(face_box, fill=(skin_r, skin_g, skin_b))

    # 4. Eyebrows
    brow_y = face_y1 + 54
    brow_c = (int(skin_r * 0.4), int(skin_g * 0.35), int(skin_b * 0.3))
    draw.arc([face_x1 + 25, brow_y - 8, face_x1 + 65, brow_y + 10], start=190, end=350, fill=brow_c, width=3)
    draw.arc([face_x2 - 65, brow_y - 8, face_x2 - 25, brow_y + 10], start=190, end=350, fill=brow_c, width=3)

    # 5. Eyes with Cornea, Iris, and Catchlights
    eye_y = brow_y + 16
    draw.ellipse([face_x1 + 30, eye_y, face_x1 + 60, eye_y + 17], fill=(250, 250, 252))
    draw.ellipse([face_x2 - 60, eye_y, face_x2 - 30, eye_y + 17], fill=(250, 250, 252))

    iris_colors = [(50, 35, 25), (40, 65, 80), (45, 75, 45), (70, 50, 30)]
    iris_c = iris_colors[rng.randint(0, len(iris_colors))]
    draw.ellipse([face_x1 + 40, eye_y + 1, face_x1 + 52, eye_y + 15], fill=iris_c)
    draw.ellipse([face_x2 - 50, eye_y + 1, face_x2 - 38, eye_y + 15], fill=iris_c)

    # Specular catchlights (natural reflection)
    draw.ellipse([face_x1 + 45, eye_y + 3, face_x1 + 48, eye_y + 6], fill=(255, 255, 255))
    draw.ellipse([face_x2 - 45, eye_y + 3, face_x2 - 42, eye_y + 6], fill=(255, 255, 255))

    # 6. Nose & Mouth
    nose_x = (face_x1 + face_x2) // 2
    nose_y = eye_y + 35
    nose_c = (int(skin_r * 0.85), int(skin_g * 0.82), int(skin_b * 0.80))
    draw.polygon([(nose_x, nose_y - 12), (nose_x - 7, nose_y + 12), (nose_x + 7, nose_y + 12)], fill=nose_c)

    mouth_y = nose_y + 32
    lip_c = (int(skin_r * 0.78), int(skin_g * 0.58), int(skin_b * 0.58))
    draw.rounded_rectangle([nose_x - 22, mouth_y, nose_x + 22, mouth_y + 14], radius=6, fill=lip_c)

    # Convert to array for pixel-level realistic transformations
    arr = np.array(img).astype(np.float32)

    # Natural camera sensor noise (present in real camera captures)
    sensor_noise = rng.normal(0, 1.8, arr.shape)
    arr = np.clip(arr + sensor_noise, 0, 255)

    if is_fake:
        # Inject Distinct Deepfake Artifact Signatures:
        # A) Blending seam artifact around facial bounding contour (pasted face boundary)
        mask = np.zeros((height, width), dtype=np.float32)
        # Create an elliptical mask for the swapped region
        y_grid, x_grid = np.ogrid[:height, :width]
        center_x, center_y = nose_x, eye_y + 18
        rx, ry = 62, 74
        dist = ((x_grid - center_x) ** 2) / (rx ** 2) + ((y_grid - center_y) ** 2) / (ry ** 2)
        
        # Boundary zone (0.85 <= dist <= 1.05) has prominent blending mismatch
        boundary = (dist >= 0.82) & (dist <= 1.05)
        arr[boundary, 0] = np.clip(arr[boundary, 0] * 1.14 + rng.randint(8, 20), 0, 255)
        arr[boundary, 1] = np.clip(arr[boundary, 1] * 0.88 - rng.randint(5, 15), 0, 255)
        arr[boundary, 2] = np.clip(arr[boundary, 2] * 1.10 + rng.randint(6, 18), 0, 255)

        # B) Frequency domain / upsampling blur mismatch inside face patch (128x128 autoencoder upscale)
        face_patch = arr[eye_y - 20 : mouth_y + 25, face_x1 + 10 : face_x2 - 10]
        patch_pil = Image.fromarray(np.clip(face_patch, 0, 255).astype(np.uint8))
        # Downscale and upscale to mimic autoencoder resolution bottleneck
        down = patch_pil.resize((patch_pil.width // 2, patch_pil.height // 2), Image.Resampling.BILINEAR)
        up = down.resize((patch_pil.width, patch_pil.height), Image.Resampling.BILINEAR)
        arr[eye_y - 20 : mouth_y + 25, face_x1 + 10 : face_x2 - 10] = np.array(up)

        # C) Eye catchlight asymmetry (corrupt catchlight in one eye)
        draw_fake = ImageDraw.Draw(img)
        arr[eye_y + 1 : eye_y + 7, face_x2 - 47 : face_x2 - 40, :] = skin_base[0] * 0.5

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Deepfake Image Dataset Preparation Tool")
    parser.add_argument("--source_dir", type=str, default=None, help="Path to raw downloaded dataset")
    parser.add_argument("--dest_dir", type=str, default=str(DATASET_DIR), help="Destination dataset folder")
    parser.add_argument("--generate_sample_data", action="store_true", help="Generate local benchmark dataset for testing")
    parser.add_argument("--sample_count", type=int, default=80, help="Number of benchmark samples per class")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set split ratio (0.0 to 1.0)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set split ratio (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")

    args = parser.parse_args()
    dest_path = Path(args.dest_dir)

    if args.generate_sample_data:
        logger.info("Triggered sample benchmark dataset generation.")
        generate_synthetic_benchmark(dest_path, samples_per_class=args.sample_count, seed=args.seed)
    elif args.source_dir:
        source_path = Path(args.source_dir)
        if not source_path.exists():
            logger.error(f"Provided source directory does not exist: {source_path}")
            sys.exit(1)
        ingest_from_source(source_path, dest_path, args.val_ratio, args.test_ratio, args.seed)
    else:
        logger.info("No source directory specified.")
        print("\n" + "=" * 70)
        print("DATASET PREPARATION INSTRUCTIONS")
        print("=" * 70)
        print("To organize your downloaded deepfake dataset, run:")
        print(f"  python prepare_dataset.py --source_dir <path_to_downloaded_images>")
        print("\nOr to generate an initial benchmark dataset to test the full pipeline:")
        print(f"  python prepare_dataset.py --generate_sample_data --sample_count 80")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
