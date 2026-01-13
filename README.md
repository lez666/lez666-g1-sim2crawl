# 🤖 G1 Sim2Crawl  
**Keyboard-Controlled Sim2Sim Crawling for Unitree G1**

<p align="center">
  <img src="sim2sim_mj/docs/images/robot_crawling_demo.gif" width="720">
</p>

**G1 Sim2Crawl** is a MuJoCo-based sim-to-sim deployment toolkit for the Unitree G1 humanoid, enabling **interactive crawling control from trained Isaac Lab policies — using only a keyboard or gamepad.**

This project makes it possible to:
- Train locomotion in Isaac Lab  
- Export policies  
- Run them in MuJoCo  
- Control the robot live using keyboard or gamepad  

No Isaac Sim runtime is required during playback.

---

## ✨ What Makes This Project Different

Compared to the original G1 Crawl pipeline, this repo adds:

- ⌨️ **Keyboard control (no gamepad required)**
- 🎮 **Full gamepad parity**
- 🔁 **Live policy switching**
- 🛡️ **Safety & gain control**
- 📦 **Plug-and-play MuJoCo sim2sim**
- 📚 **Bilingual documentation**
- 🎥 **Reproducible demo pipeline**

This turns a research policy into a **hands-on, testable, reproducible robotics system**.

---

## 🧬 Project Lineage

This project is derived from and builds upon:

> **jloganolson/g1_crawl**  
> https://github.com/jloganolson/g1_crawl

lez666-g1-sim2crawl extends the original project with:
- MuJoCo-based sim2sim deployment
- Keyboard-based control (no gamepad required)
- Improved documentation and reproducibility

All credit for the original G1 crawling policy and Isaac Lab training framework belongs to the original authors.

---

## 🚀 Quick Start (Keyboard Sim2Sim)

### Option 1: Automatic Installation (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd lez666-g1-sim2crawl

# Run the automatic setup script
./setup.sh

# Activate environment and run
conda activate g1-crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

### Option 2: Manual Installation

```bash
# Create conda environment
conda create -n g1-crawl python=3.10 -y
conda activate g1-crawl

# Install dependencies
pip install -r sim2sim_mj/requirements.txt

# Run simulation
python sim2sim_mj/run_sim2sim_keyboard.py
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md)

No controller required — just use your keyboard.

---

## ⌨️ Default Key Bindings

| Action | Keys |
|--------|------|
| Forward / Backward | ↑ / ↓ |
| Strafe Left / Right | ← / → |
| Rotate Left / Right | Z / C |
| Mode | I (stand) · J (damped) · K (crawl) |
| Switch Policy | SPACE |
| Gain Up / Down | Q / A |
| Print Gains | H |
| Exit | ESC |

---

## 📘 Keyboard Control Guide

- **English** → `sim2sim_mj/README_keyboard_en.md`  
- **中文 (Chinese)** → `sim2sim_mj/README_keyboard_zh.md`

---

## 🎮 Gamepad Mode

You can also use a standard controller:

```bash
python sim2sim_mj/run_sim2sim.py
```

Supports:

- Analog velocity  
- Policy cycling  
- Mode switching  
- Safety exit  
- CSV diagnostics  

---

## 🧪 Training (Isaac Lab)

### Minimal headless training

```bash
python scripts/rsl_rl/train.py --task g1-crawl --headless
```

## 🧪 Play a Trained Policy

```bash
python scripts/rsl_rl/play.py --task g1-crawl --headless --video --enable_cameras
```

## 🤖 Real Robot Deployment

The `deployment/` folder contains robot-side execution pipelines with:

- PD control  
- Watchdogs  
- Hardware interfaces  

> ⚠ Always validate in simulation before deploying on real hardware.

---

## 📂 Key Paths

| Purpose | Path |
|--------|------|
| Training logs | `logs/rsl_rl/` |
| Sweep results | `sweep-logs/` |
| Sim2Sim policies | `sim2sim_mj/policies/` |
| MuJoCo model | `sim2sim_mj/scene.xml` |

---

## 🛠 3D Printed Parts

- **Head**: Onshape link  
- **Arms**: Onshape link  
- **Print files**: `3d-printed-parts/`
