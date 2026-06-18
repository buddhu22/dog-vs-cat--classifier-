from pathlib import Path
from typing import Tuple

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / 'models' / 'best_model.keras'

# Dataset paths
TRAIN_DIR = BASE_DIR / 'catsvsdogs' / 'train'
TEST_DIR = BASE_DIR / 'catsvsdogs' / 'test'

# Output directories
OUTPUTS_DIR = BASE_DIR / 'outputs'
ASSETS_DIR = BASE_DIR / 'assets'

# Model parameters
# CRITICAL: This must match the image_size used in train.py exactly.
IMAGE_SIZE: Tuple[int, int] = (256, 256)
INPUT_SHAPE: Tuple[int, int, int] = (256, 256, 3)

# Class labels — order MUST match image_dataset_from_directory's alphabetical sort.
# Subdirectories: cats/ dogs/ → sorted → ['cats', 'dogs'] → label 0 = cats, label 1 = dogs
CLASSES = ['Cat', 'Dog']

# Ensure necessary directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'models').mkdir(parents=True, exist_ok=True)
