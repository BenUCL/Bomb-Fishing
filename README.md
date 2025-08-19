# Bomb Fishing Classifier

A machine learning pipeline for detecting explosive fishing using underwater audio recordings.

## 🚀 Quick Start

### For Model Retraining (New Data)
```bash
# Complete retraining pipeline with new data (UV automatically manages dependencies)
cd retraining_scripts
uv run python run_complete_pipeline.py
```
See [retraining_scripts/README.md](retraining_scripts/README.md) for detailed instructions.

### For Inference Only

#### New .keras Models (Recommended)
```bash
# Run inference on new audio files (UV handles environment automatically)
uv run python simple_inference.py models/retrained_best_model.keras test_file.wav
uv run python modern_inference.py  # batch processing
```

#### Legacy model/ Directory Models
```bash
# For existing models in directory format - requires conda environment
conda activate conda-bomb-env  # using set_up/linux_env.yml
python simple_inference.py code/model test_file.wav
cd code && python -m scripts.batch_runner  # legacy batch processing
```

See [INFERENCE_INSTRUCTIONS.md](INFERENCE_INSTRUCTIONS.md) for detailed inference guide.

## 📁 Project Structure

```
Bomb-Fishing/
├── retraining_scripts/          # 🔄 Complete retraining pipeline (creates .keras models)
│   ├── README.md               # Detailed retraining guide
│   ├── QUICK_START.md          # 5-minute setup guide
│   ├── run_complete_pipeline.py # One-click pipeline runner
│   ├── eval_model.py           # Model evaluation (new + legacy)
│   ├── tune_threshold.py       # Threshold optimization (new + legacy)
│   ├── parent_script.py        # Legacy batch inference
│   └── child_script.py         # Legacy batch inference worker
├── code/                       # 🔧 Core inference infrastructure (legacy)
│   └── model/                  # 📁 Legacy model directory format
├── models/                     # 🤖 New trained models (.keras files)
├── simple_inference.py         # 🎯 Single file inference (new + legacy)
├── modern_inference.py         # 📦 Modern batch inference (.keras models)
└── set_up/                     # 🛠️ Environment configurations
    └── linux_env.yml          # Required for legacy models
```

## 🛠️ Environment Setup

### Option 1: UV with pyproject.toml (Recommended - Zero Config!)
```bash
# UV automatically manages dependencies from pyproject.toml
# No virtual environment setup needed!
uv run python --version  # Test that UV works

# Dependencies are automatically installed when you run scripts
uv run python simple_inference.py --help
```

### Option 2: UV with Manual Environment
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements_new_stack.txt
```

### Option 3: Automated UV Setup
```bash
./rebuild_environment_uv.sh
```

### Option 4: Conda (Traditional)
```bash
conda env create -f set_up/macos_arm64_env_fixed_new.yml
conda activate bomb-audio-env-arm64-new
```

## 📖 Documentation

- **[Retraining Guide](retraining_scripts/README.md)** - Complete model retraining pipeline (creates new .keras models)
- **[Quick Start](retraining_scripts/QUICK_START.md)** - 5-minute setup for retraining
- **[Inference Guide](INFERENCE_INSTRUCTIONS.md)** - Run inference on audio files (new + legacy models)
- **[Environment Setup](UV_ENVIRONMENT_REBUILD_README.md)** - Modern UV setup for new models

## 🔄 Model Formats

### New .keras Models (Recommended)
- **Format**: Single `.keras` file
- **Environment**: UV with modern Python dependencies
- **Created by**: Retraining pipeline in `retraining_scripts/`
- **Usage**: `uv run python script.py model.keras`

### Legacy model/ Directory Models  
- **Format**: Directory with `variables/`, `saved_model.pb`, `keras_metadata.pb`
- **Environment**: Conda with `set_up/linux_env.yml`
- **Location**: `code/model/` (existing)
- **Usage**: `conda activate conda-bomb-env && python script.py model/`

## Legacy Setup (Linux)
```
cd ~/bomb_fishing/set_up
conda env create -f linux_env.yml -n conda-bomb-env
conda activate conda-bomb-env
```

## Legacy Inference
Adjust the filepaths at the top of ~/bomb_fishing/code/config.py and run:
```
cd ~/bomb_fishing/code
python -m scripts.batch_runner
```

## Create an env for open soundscape
```
conda create -n open_soundscape_env \
  python=3.9 \
  jupyterlab \
  geopandas \
  contextily \
  scipy \
  numpy \
  pandas \
  shapely \
  -c conda-forge

conda activate open_soundscape_env
pip install opensoundscape
conda install ipykernel notebook jupyter -y

# Make kernel visible to jupyter notebooks
python -m ipykernel install --user --name open_soundscape_env --display-name "Python (open_soundscape_env)"
```


