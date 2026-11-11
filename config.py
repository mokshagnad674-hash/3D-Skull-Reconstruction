import os

DATA_ROOT = r"d:\ML_DATASET"

CT_DIR = os.path.join(DATA_ROOT, "CT")
MASK_DIR = os.path.join(DATA_ROOT, "mask")

BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
