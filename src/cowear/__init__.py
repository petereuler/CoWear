"""Auditable CoWear paper reproduction package."""

__version__ = "1.0.1"
DATASET_URL = "https://huggingface.co/datasets/zyshe/CoWear"
DATASET_VERSION = "CoWear-3device-363-v1"
PROTOCOL_VERSION = "paper-v1"

# Names used in the manuscript are deliberately kept separate from the
# storage-role names used by the released Hugging Face files.
PAPER_ROLES = ("phone", "watch", "glasses")
STORAGE_ROLE_FOR_PAPER = {"phone": "mobile", "watch": "watch", "glasses": "rokid"}
PAPER_ROLE_FOR_STORAGE = {value: key for key, value in STORAGE_ROLE_FOR_PAPER.items()}
# CoWear LSTM input is the device-provided signal: gyro and accelerometer.
# Linear acceleration is derived for event detection and diagnostics only.
INPUT_FEATURES = 6
MODEL_HIDDEN_SIZE = 128
TRAIN_BATCH_SIZE = 1024
TRAIN_MAX_EPOCHS = 100

__all__ = [
    "DATASET_URL", "DATASET_VERSION", "PROTOCOL_VERSION", "PAPER_ROLES",
    "STORAGE_ROLE_FOR_PAPER", "PAPER_ROLE_FOR_STORAGE", "INPUT_FEATURES",
    "MODEL_HIDDEN_SIZE", "TRAIN_BATCH_SIZE", "TRAIN_MAX_EPOCHS", "__version__",
]
