#!/usr/bin/env python3
"""
Local configuration override for the current directory structure
This works with the existing code structure in /Users/sonnyburniston/Bomb-Fishing/
"""

import os
from pathlib import Path

# Current directory structure
BASE_DIR = os.getcwd()  # /Users/sonnyburniston/Bomb-Fishing
PROJECT_DIR = Path(BASE_DIR)
CODE_DIR = PROJECT_DIR / "code"
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = CODE_DIR / "model"
SCRATCH_DIR = PROJECT_DIR / "scratch"

# Audio processing settings (matching the inference pipeline)
BATCH_SIZE = 100
SAMPLE_RATE = 8_000
WINDOW_LENGTH_SEC = 2.88
STAGGER_SEC = WINDOW_LENGTH_SEC / 2

# Input/Output paths for our processed audio
RAW_AUDIO = str(DATA_DIR / "raw_audio" / "final_dataset")
OUTPUT_FOLDER = "detections"
INPUT_DIR = RAW_AUDIO
OUTPUT_DIR = DATA_DIR / OUTPUT_FOLDER

# Create directories if they don't exist
for directory in [DATA_DIR, SCRATCH_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

print(f"Local config loaded for: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Data directory: {DATA_DIR}")
