# Bomb-Fishing: GPU-enabled setup guide

This guide explains how to get a conda env running TensorFlow 2.10 with GPU support on Ubuntu 24.04 (RTX 4060 tested).

---

## 1. Clone the repo

```bash
git clone https://github.com/BenUCL/Bomb-Fishing.git
cd Bomb-Fishing
```

---

## 2. Create & activate the Conda environment

We’ve already got a cleaned-up YAML (`bomb-detector-linux.yml`) in `set_up`. Run:

```bash
conda env create -f set_up/linux_env.yml
conda activate conda-bomb-env
```

---

## 3. Install CUDA Toolkit & cuDNN

TensorFlow 2.10 is built against CUDA 11.2 + cuDNN 8.1. Pull them into your env:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

---

## 4. Add an `activate.d` hook

Create a tiny shell script so your env picks up both its own libraries **and** the system driver:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat << 'EOF' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Point to the conda env’s CUDA install (optional helper)
export CUDA_HOME="$CONDA_PREFIX"

# 1) Prepend env libs for cudatoolkit & cuDNN  
# 2) Include system path so libcuda.so.1 (driver) can be found  
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib":/usr/lib/x86_64-linux-gnu:"$LD_LIBRARY_PATH"

# Ensure conda-installed binaries (nvcc, etc) are on PATH
export PATH="$CONDA_PREFIX/bin":"$PATH"
EOF
```

Then reload:

```bash
conda deactivate
conda activate conda-bomb-env
```

---

## 5. (Optional) Install `nvcc`

If you need the CUDA compiler:

```bash
# either via conda:
conda install -c conda-forge cudatoolkit-dev

# or via apt (system-wide):
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

---

## 6. Verify everything

1. **Check driver**  
   ```bash
   nvidia-smi
   ```
   You should see your RTX 4060 listed.

2. **Check TF sees your GPU**  
   ```bash
   python - << 'EOF'
   import tensorflow as tf
   print("TF version:", tf.__version__)
   print("GPUs found:", tf.config.list_physical_devices("GPU"))
   EOF
   ```
   You should see something like:
   ```
   TF version: 2.10.1
   GPUs found: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```

---

## Quick troubleshooting

- **Mismatch errors** → adjust `cudatoolkit=` and `cudnn=` to the versions TensorFlow expects (see TensorFlow’s install guide).
- **Missing `libcuda.so.1`** → ensure `/usr/lib/x86_64-linux-gnu/libcuda.so.1` exists (reinstall driver with `sudo apt install --reinstall nvidia-driver-535`).
- **Global vs env tweaks** → we only changed the conda env; your `~/.bashrc` remains untouched.

---

That’s it! Push this README to your repo and you’ll have a clear, self-contained setup for anyone to follow.
