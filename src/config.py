from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "iivp-2026-challenge"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
RUN_DIR = EXPERIMENT_DIR / "runs"
SUBMISSION_DIR = ROOT_DIR / "submissions"
REPORT_DIR = EXPERIMENT_DIR / "reports"
REPORT_NAME = "scientific_comparison"

SEED = 42
IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001
VAL_RATIO = 0.1
NUM_WORKERS = 0

MODEL_NAME = "deotte_cnn"
MODEL_CHOICES = [
    "simple_cnn",
    "improved_cnn",
    "strided_cnn",
    "deotte_cnn",
]

DROPOUT = 0.35
WEIGHT_DECAY = 0.0005
OPTIMIZER = "adamw"
OPTIMIZER_CHOICES = ["adam", "adamw"]
SCHEDULER = "cosine"
SCHEDULER_CHOICES = ["none", "step_decay", "cosine"]
LR_GAMMA = 0.95
LABEL_SMOOTHING = 0.0
TTA = False

NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
FOREGROUND_THRESHOLD = 20
CENTER_MARGIN = 4
PREPROCESS_PROFILE = "centered"
PREPROCESS_PROFILE_CHOICES = ["plain", "centered"]

# Affine/bootstrap settings. translate is a fraction of image width/height;
# 0.08 means roughly +/-3 pixels for a 32x32 image.
AFFINE_SETTINGS = {
    "none": {
        "description": "No synthetic affine samples are added.",
        "copies_per_original": 0,
        "degrees": 0,
        "translate": (0.0, 0.0),
        "scale": (1.0, 1.0),
        "shear": 0.0,
    },
    "mild": {
        "description": "Small random rotation, translation, and scale changes. Intended to preserve digit identity while improving robustness to alignment noise.",
        "copies_per_original": 1,
        "degrees": 8,
        "translate": (0.08, 0.08),
        "scale": (0.92, 1.08),
        "shear": 0.0,
    },
    "strong": {
        "description": "Larger affine perturbations including shear. Useful only if validation images contain strong distortion; otherwise it may create unrealistic samples.",
        "copies_per_original": 2,
        "degrees": 14,
        "translate": (0.12, 0.12),
        "scale": (0.86, 1.16),
        "shear": 8.0,
    },
}

# The experiment grid. train_experiments.py trains every model x every data setting once.
REPORT_MODELS = MODEL_CHOICES
REPORT_DATA_SETTINGS = [
    {"name": "plain", "preprocess_profile": "plain", "affine_setting": "none"},
    {"name": "centered", "preprocess_profile": "centered", "affine_setting": "none"},
    {"name": "centered_mild_bootstrap", "preprocess_profile": "centered", "affine_setting": "mild"},
    {"name": "centered_strong_bootstrap", "preprocess_profile": "centered", "affine_setting": "strong"},
]

MAKE_TEST_SUBMISSIONS = False
DETAILED_CLASS_FILES = False
MAX_MISCLASSIFIED_IMAGES = 24
N_PREPROCESS_EXAMPLES = 6
N_BOOTSTRAP_EXAMPLES = 6
INFERENCE_TIMING_REPEATS = 2
