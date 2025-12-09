"""
Configuration settings for the Adaptive Weather Attacks project.

This module provides centralized configuration for:
- Paths and directories
- Dataset settings
- Model architectures
- Training hyperparameters
- Attack parameters
- Diffusion/V-CFG settings
"""

import os
import torch
from pathlib import Path


# =============================================================================
# AUTO-DETECT ENVIRONMENT (Colab vs Local)
# =============================================================================

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

if IN_COLAB:
    # Colab paths
    PROJECT_ROOT = Path("/content/adaptive-weather-attacks")
    DATA_DIR = Path("/content/GTSRB_dataset")
else:
    # Local paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "GTSRB_dataset"

# Derived paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset structure
GTSRB_TRAIN_DIR = DATA_DIR / "GTSRB_final_training_images"
GTSRB_TEST_DIR = DATA_DIR / "GTSRB_final_test_images"
GTSRB_TEST_CSV = DATA_DIR / "GTSRB_Final_Test_GT" / "GT-final_test.csv"


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for dir_path in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OUTPUTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if IN_COLAB else 0  # Colab has multiple CPUs
PIN_MEMORY = torch.cuda.is_available()


# =============================================================================
# DATASET SETTINGS
# =============================================================================

NUM_CLASSES = 43
IMAGE_SIZE = 224
VAL_SPLIT = 0.1

# ImageNet normalization (used by pretrained models)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =============================================================================
# MODEL SETTINGS
# =============================================================================

AVAILABLE_MODELS = ["resnet50", "efficientnet_b0", "vit"]


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001


# =============================================================================
# ADVERSARIAL ATTACK SETTINGS
# =============================================================================

# FGSM
FGSM_EPS = 0.03

# PGD
PGD_EPS = 0.03
PGD_ALPHA = 0.01
PGD_STEPS = 10

# C&W
CW_C = 1
CW_KAPPA = 0
CW_STEPS = 1000
CW_LR = 0.01


# =============================================================================
# DIFFUSION SETTINGS
# =============================================================================

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 30
DIFFUSION_STRENGTH = 0.5


# =============================================================================
# VARIABLE CFG SETTINGS (Main Contribution)
# =============================================================================

class CFGSchedule:
    """Enum-like class for CFG schedule types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    INVERSE = "inverse"


# Default CFG parameters
CFG_CONSTANT_SCALE = 7.5
CFG_START_SCALE = 12.0
CFG_END_SCALE = 3.0
CFG_STEP_TRANSITION = 0.5  # Fraction of total steps


# =============================================================================
# WEATHER PROMPTS
# =============================================================================

WEATHER_PROMPTS = {
    "fog": [
        "a traffic sign in dense fog",
        "a traffic sign in thick fog",
        "a traffic sign in heavy fog with low visibility",
    ],
    "rain": [
        "a traffic sign in heavy rain",
        "a traffic sign in rainstorm with wet reflections",
        "a traffic sign in rain at dusk",
    ],
    "snow": [
        "a traffic sign in thick snow",
        "a traffic sign in blizzard conditions",
        "a traffic sign covered in snow",
    ],
    "night": [
        "a traffic sign at night",
        "a traffic sign at night under streetlights",
        "a traffic sign illuminated by headlights at night",
    ],
    "glare": [
        "a traffic sign in harsh sunlight with glare",
        "a traffic sign with lens flare",
        "a traffic sign in bright sunlight, overexposed",
    ],
}


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

DEFAULT_NUM_SAMPLES = 100
RANDOM_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_path(model_name: str) -> Path:
    """Get the checkpoint path for a given model."""
    return CHECKPOINT_DIR / f"{model_name}_best.pth"


def print_config():
    """Print current configuration settings."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"Device: {DEVICE}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    print("-" * 60)
    print(f"Num Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("-" * 60)
    print(f"Diffusion Model: {DIFFUSION_MODEL_ID}")
    print(f"Inference Steps: {NUM_INFERENCE_STEPS}")
    print(f"CFG Start Scale: {CFG_START_SCALE}")
    print(f"CFG End Scale: {CFG_END_SCALE}")
    print("=" * 60)


# Create directories on import
ensure_dirs()
