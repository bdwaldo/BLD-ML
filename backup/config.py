# config.py

import torch

# === Data & augmentation params ===
BATCH_SIZE   = 16
ROT_DEG      = 90
NOISE_STD    = 0.05
IMG_SIZE     = 384

# Number of DataLoader workers and pinning behaviour
NUM_WORKERS  = 2
PIN_MEMORY   = torch.cuda.is_available()

# Normalization constants for ImageNet / EfficientNet‑V2
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional: default data root (can be overridden via CLI args)
DATA_ROOT = "/90daydata/nematode_ml/BLD/NematodeDataset"
