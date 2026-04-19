# Wii-RL Installation Guide

This README focuses on installation and setup across Windows, Linux, and macOS.

---

## 1) Prerequisites

### Windows
- **Python 3.12** (required by the Dolphin scripting fork)  
  https://www.python.org/downloads/release/python-3120/
- **Visual Studio C++ Build Tools**  
  Install "Desktop development with C++" from https://visualstudio.microsoft.com/downloads/

### Linux
- **Python 3.12+** installed system-wide (virtual env version should match system Python)

### macOS
- **Python 3.12+**
- For the prebuilt Dolphin download, install **Python 3.13** via Homebrew:
  ```sh
  brew update
  brew install python@3.13
  ```
  If you use a virtual env, match the Homebrew Python version.

---

## 2) Clone the Repository
```sh
git clone https://github.com/LJH-coding/DRL-HW4-Bonus.git
cd DRL-HW4-Bonus
```

---

## 3) Game ROM
Place the Mario Kart Wii ROM here:
- Path: `game/mkw.iso`
- Region: **European RMCP01**
- MD5: `e7b1ff1fabb0789482ce2cb0661d986e`

We cannot distribute ROMs. You must provide your own copy.

---

## 4) Install Python Dependencies
GPU (CUDA) build:
```sh
pip install -r requirements.txt
```

CPU-only build:
```sh
pip install -r requirements_cpu.txt
```

---

## 5) Docker (Linux, Recommended for Headless)
Docker avoids building Dolphin on the host. The image builds Dolphin once and
uses Xvfb for headless execution.

Build the image:
```sh
docker build -t wii-rl .
```

Build a CPU-only image:
```sh
docker build -t wii-rl-cpu --build-arg TORCH_REQUIREMENTS=requirements_cpu.txt .
```

Run training (CPU):
```sh
docker run --rm -it --shm-size=1g \
  -v "$(pwd)/game:/workspace/game" \
  -v "$(pwd)/MarioKartSaveStates:/workspace/MarioKartSaveStates" \
  -v "$(pwd)/models:/workspace/models" \
  wii-rl
```

Run training with NVIDIA GPU (requires NVIDIA Container Toolkit):
```sh
docker run --rm -it --gpus all --shm-size=1g \
  -v "$(pwd)/game:/workspace/game" \
  -v "$(pwd)/MarioKartSaveStates:/workspace/MarioKartSaveStates" \
  -v "$(pwd)/models:/workspace/models" \
  wii-rl
```

Run a custom command:
```sh
docker run --rm -it --shm-size=1g \
  -v "$(pwd)/game:/workspace/game" \
  -v "$(pwd)/MarioKartSaveStates:/workspace/MarioKartSaveStates" \
  -v "$(pwd)/models:/workspace/models" \
  wii-rl python evaluate.py --model_path /workspace/models/model.pt
```

Note: The Docker image uses GPU-enabled PyTorch by default. Use the CPU-only
build arg if you do not have NVIDIA drivers or GPU access.

---

## 6) Install Dolphin (Windows / macOS)
```sh
python3 scripts/download_dolphin.py
```

### Linux (build from source)
If you do not have a prebuilt release:
```sh
bash scripts/build-dolphin-linux.sh
```
The build script pins the Dolphin fork to a known working tag for scripting.

---

## 7) Clone Dolphin Instances
This creates `dolphin1`, `dolphin2`, ... for parallel training.
```sh
python3 scripts/clone_dolphins.py --clones 3
```

---

## 8) Download Save States
```sh
python3 scripts/download_savestates.py
```

---

## 9) Headless Linux (Server / No Display)
Use Xvfb to provide a virtual display:
```sh
sudo apt-get install -y xvfb
bash scripts/run_headless_linux.sh
```

To run a custom command:
```sh
bash scripts/run_headless_linux.sh python training.py
```

---
