from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "iivp-2026-challenge"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
SUBMISSION_DIR = ROOT_DIR / "submissions"

SEED = 42
IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
VAL_RATIO = 0.1
NUM_WORKERS = 0
