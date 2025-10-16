# G1 Crawl - Standalone Deployment Package

This package contains everything needed to run G1 crawl policies without mjlab.

## Contents

- `deploy_standalone.py` - Deployment script with gamepad support
- `scene.xml` - MuJoCo scene (robot + ground)
- `meshes/` - Robot mesh files (37 STL files)
- `policies/` - Trained policy files
- `poses/` - Initial pose configurations

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### With Gamepad (Recommended)

1. Connect a standard gamepad (Xbox, PlayStation, etc.)
2. Edit `CONFIG` section in `deploy_standalone.py` to customize settings
3. Run:
```bash
python deploy_standalone.py
```

**Gamepad Controls:**
- **Left Stick Y**: Forward/backward velocity
- **Right Stick X**: Rotation (angular velocity)
- **A Button**: Switch to `policy_crawl.pt`
- **B Button**: Switch to `policy_shamble.pt`
- **X Button**: Switch to `policy_crawl_start.pt`
- **Y Button**: Switch to `policy_shamble_prev.pt`

### Customizing Button Mapping

Edit the `button_policy_map` in the `CONFIG` section:
```python
"button_policy_map": {
    0: "policies/policy_crawl.pt",      # A button
    1: "policies/policy_shamble.pt",    # B button
    2: "policies/policy_crawl_start.pt", # X button
    3: "policies/policy_shamble_prev.pt", # Y button
},
```

### Without Gamepad

Set `use_gamepad: False` in CONFIG. Robot will hold default pose without movement commands.

## Configuration

All settings are in the `CONFIG` dictionary at the top of `deploy_standalone.py`:

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
- Standard gamepad (optional, for real-time control)

## Package Info

Created from: mjlab-g1-crawl
Policies included: 4
- `policy_crawl.pt`
- `policy_shamble.pt`
- `policy_crawl_start.pt`
- `policy_shamble_prev.pt`
