# G1 Crawl - Standalone Deployment Package

This package contains everything needed to run G1 crawl policies without mjlab.

## 🎯 Enhanced Features

This version includes **keyboard control support** as an alternative to gamepad control, making the simulation accessible without requiring a gamepad.

### Main Contributions

1. **Complete Keyboard Control Implementation**
   - Added `run_sim2sim_keyboard.py` with full keyboard control support
   - Implemented cross-platform keyboard input handling using `pynput` library
   - Supports all original gamepad functions including movement control, mode switching, policy switching, and gain adjustment

2. **Keyboard Mapping Design**
   - **Arrow Keys**: Forward/backward and lateral movement control
   - **IJK Keys**: Mode switching (I=stand, J=damped, K=crawl)
   - **Z/C Keys**: Rotation control
   - **SPACE Key**: Policy cycling
   - **Q/A/H Keys**: Real-time gain adjustment
   - **ESC Key**: Exit program

3. **Full Feature Parity**
   - Completely preserves all original control logic and safety features
   - Supports all control modes (default_pos, crawl_pos, damped, policy)
   - Maintains complete safety monitoring system

4. **Comprehensive Documentation**
   - Detailed Chinese and English READMEs with keyboard control instructions
   - Feature comparison table between keyboard and gamepad controls

For detailed keyboard control instructions, see:
- [README_keyboard_en.md](README_keyboard_en.md) (English)
- [README_keyboard_zh.md](README_keyboard_zh.md) (Chinese)

## Contents

- `run_sim2sim.py` - Original deployment script with gamepad support
- `run_sim2sim_keyboard.py` - **NEW**: Deployment script with keyboard control support
- `scene.xml` - MuJoCo scene (robot + ground)
- `meshes/` - Robot mesh files (37 STL files)
- `policies/` - Trained policy files
- `poses/` - Initial pose configurations
- `README_keyboard_en.md` - **NEW**: Detailed keyboard control guide (English)
- `README_keyboard_zh.md` - **NEW**: Detailed keyboard control guide (Chinese)

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### With Keyboard (Recommended for users without gamepad)

1. Run the keyboard control script:
```bash
conda activate g1-crawl
python run_sim2sim_keyboard.py
```

**Keyboard Controls:**
- **Arrow Keys ↑↓**: Forward/backward movement
- **Arrow Keys ←→**: Strafe left/right
- **Z/C Keys**: Rotate left/right
- **I Key**: Switch to default position mode (stand)
- **J Key**: Switch to damped mode
- **K Key**: Switch to crawl position mode
- **SPACE Key**: Cycle through policies (only in policy mode)
- **Q/A Keys**: Increase/decrease gain multipliers
- **H Key**: Print current gain values
- **ESC Key**: Exit program

The robot starts in standing state (policy mode) and is ready for immediate control.

For detailed keyboard control instructions, see [README_keyboard_en.md](README_keyboard_en.md) or [README_keyboard_zh.md](README_keyboard_zh.md).

### With Gamepad (Original Method)

1. Connect a standard gamepad (Xbox, PlayStation, etc.)
2. Edit `CONFIG` section in `run_sim2sim.py` to customize settings
3. Run:
```bash
conda activate g1-crawl
python run_sim2sim.py
```

**Gamepad Controls:**
- **Left Stick Y**: Forward/backward velocity
- **Left Stick X**: Strafe left/right
- **Right Stick X**: Rotation (angular velocity)
- **D-PAD**: Mode switching (UP=stand, DOWN=crawl, LEFT=damped, RIGHT=policy)
- **A Button**: Cycle through policies
- **START Button**: Enable system
- **SELECT Button**: Emergency stop / Exit

### Customizing Button Mapping

Edit the `button_policy_map` in the `CONFIG` section of `run_sim2sim.py`:
```python
"button_policy_map": {
    0: "policies/policy_crawl.pt",      # A button
    1: "policies/policy_shamble.pt",    # B button
    2: "policies/policy_crawl_start.pt", # X button
    3: "policies/policy_shamble_prev.pt", # Y button
},
```

## Configuration

### Keyboard Control (`run_sim2sim_keyboard.py`)

All settings are in the `CONFIG` dictionary at the top of the script:

- `model_xml`: Path to MJCF scene file
- `policy_path`: Initial policy to load
- `action_scale`: Global action multiplier (0-1)
- `n_substeps`: Simulation substeps per policy step
- `init_pose_json`: Initial robot pose
- `use_keyboard`: Enable keyboard control (default: True)
- `gain_step`: Step size for gain adjustment (default: 0.1)
- `max_lin_vel`: Maximum forward velocity (m/s)
- `max_ang_vel`: Maximum angular velocity (rad/s)

### Gamepad Control (`run_sim2sim.py`)

All settings are in the `CONFIG` dictionary at the top of the script:

- `model_xml`: Path to MJCF scene file
- `policy_path`: Initial policy to load
- `action_scale`: Global action multiplier (0-1)
- `n_substeps`: Simulation substeps per policy step
- `init_pose_json`: Initial robot pose
- `use_gamepad`: Enable/disable gamepad control
- `max_lin_vel`: Maximum forward velocity (m/s)
- `max_ang_vel`: Maximum angular velocity (rad/s)
- `button_policy_map`: Face button to policy mapping

## System Requirements

- Python 3.10+
- CPU only (no GPU required)
- Linux/macOS/Windows
- Standard gamepad (optional, for gamepad control)
- Keyboard (for keyboard control, no additional hardware needed)

### Additional Dependencies for Keyboard Control

The keyboard control script requires `pynput` library:
```bash
pip install pynput
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Package Info

Created from: mjlab-g1-crawl
Policies included: 4
- `policy_crawl.pt`
- `policy_shamble.pt`
- `policy_crawl_start.pt`
- `policy_shamble_prev.pt`
