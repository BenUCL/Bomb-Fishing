# config.py

import os
from pathlib import Path

# Give path to the raw audio files 
RAW_AUDIO = "/media/bwilliams/sda1/four_islands_bombs/M36_pulau_pala"

# Outputs will be saved in data/detections/<raw_name>, where <raw_name> is 
# the name of the folder containing the raw audio files
OUTPUT_FOLDER = "detections"  

# Set base directory to hide paths
BASE_DIR = os.getenv("BASE_DIR")
if not BASE_DIR:
    raise ValueError("BASE_DIR environment variable is not set.")

# Set up paths
PROJECT_DIR = Path(BASE_DIR) / "bomb_fishing"
CODE_DIR    = PROJECT_DIR / "code"
DATA_DIR    = PROJECT_DIR / "data"
INPUT_DIR   = RAW_AUDIO      
OUTPUT_DIR  = DATA_DIR / OUTPUT_FOLDER  
OUTPUT_FOLDER = "detections" 
MODEL_DIR   = CODE_DIR / "model"
SCRATCH_DIR = PROJECT_DIR / "scratch" # where tmp files get stored

# Other params
BATCH_SIZE = 100
SAMPLE_RATE = 8_000
WINDOW_LENGTH_SEC = 2.88
STAGGER_SEC = WINDOW_LENGTH_SEC / 2

# Print the paths for verification
for name, val in [
    ("BASE_DIR", BASE_DIR),
    ("PROJECT_DIR", PROJECT_DIR),
    ("CODE_DIR", CODE_DIR),
    ("DATA_DIR", DATA_DIR),
    ("INPUT_DIR", INPUT_DIR),
    ("OUTPUT_DIR", OUTPUT_DIR),
    ("MODEL_DIR", MODEL_DIR),
]:
    print(f"{name}: {val}")
print()
