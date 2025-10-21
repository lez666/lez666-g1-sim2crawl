from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

# Joint limits from deployment/utils.py RESTRICTED_JOINT_RANGE (in MuJoCo joint order)
RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg. 6
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
)

# Joint names in MuJoCo order from deployment/utils.py
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


class SimpleSceneCfg(InteractiveSceneCfg):
    """Minimal scene with just the robot."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    Robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = SimpleSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Get robot
    robot = scene["Robot"]
    
    # Get joint information
    joint_names = robot.data.joint_names
    soft_limits = robot.data.soft_joint_pos_limits[0]  # [num_joints, 2] (min, max)
    
    # Try to get hard limits too
    try:
        hard_limits = robot.data.joint_pos_limits[0]
    except:
        hard_limits = None
    
    print("=" * 120)
    print("JOINT LIMITS COMPARISON: Isaac Lab vs Real Robot (utils.py)")
    print("=" * 120)
    print(f"soft_joint_pos_limit_factor in g1.py: 0.9")
    print()
    
    # Print header
    print(f"{'Index':<6} {'Joint Name':<30} {'Isaac Soft Min':<15} {'Isaac Soft Max':<15} {'Real Min':<12} {'Real Max':<12} {'Match?':<8}")
    print("-" * 120)
    
    # Build mapping from joint name to index in RESTRICTED_JOINT_RANGE
    # RESTRICTED_JOINT_RANGE is in MuJoCo joint order (same as MUJOCO_JOINT_NAMES)
    mujoco_name_to_idx = {name: i for i, name in enumerate(MUJOCO_JOINT_NAMES)}
    
    # Track mismatches
    mismatches = []
    
    for i, joint_name in enumerate(joint_names):
        soft_min = float(soft_limits[i, 0])
        soft_max = float(soft_limits[i, 1])
        
        # Get corresponding real robot limits
        if joint_name in mujoco_name_to_idx:
            real_idx = mujoco_name_to_idx[joint_name]
            real_min, real_max = RESTRICTED_JOINT_RANGE[real_idx]
            
            # Check if they match (with small tolerance for floating point)
            tolerance = 0.001
            min_match = abs(soft_min - real_min) < tolerance
            max_match = abs(soft_max - real_max) < tolerance
            match = "✓" if (min_match and max_match) else "✗"
            
            if not (min_match and max_match):
                mismatches.append({
                    'joint': joint_name,
                    'isaac_min': soft_min,
                    'isaac_max': soft_max,
                    'real_min': real_min,
                    'real_max': real_max,
                    'min_diff': soft_min - real_min,
                    'max_diff': soft_max - real_max,
                })
            
            print(f"{i:<6} {joint_name:<30} {soft_min:<15.6f} {soft_max:<15.6f} {real_min:<12.6f} {real_max:<12.6f} {match:<8}")
        else:
            print(f"{i:<6} {joint_name:<30} {soft_min:<15.6f} {soft_max:<15.6f} {'N/A':<12} {'N/A':<12} {'?':<8}")
    
    print("-" * 120)
    
    if hard_limits is not None:
        print("\nHard limits (before soft factor applied):")
        print(f"{'Index':<6} {'Joint Name':<30} {'Hard Min':<15} {'Hard Max':<15}")
        print("-" * 80)
        for i, joint_name in enumerate(joint_names):
            hard_min = float(hard_limits[i, 0])
            hard_max = float(hard_limits[i, 1])
            print(f"{i:<6} {joint_name:<30} {hard_min:<15.6f} {hard_max:<15.6f}")
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    if len(mismatches) == 0:
        print("✓ ALL JOINT LIMITS MATCH between Isaac Lab (soft) and Real Robot (utils.py)")
    else:
        print(f"✗ FOUND {len(mismatches)} MISMATCHES:")
        print()
        for m in mismatches:
            print(f"Joint: {m['joint']}")
            print(f"  Isaac soft: [{m['isaac_min']:.6f}, {m['isaac_max']:.6f}]")
            print(f"  Real robot: [{m['real_min']:.6f}, {m['real_max']:.6f}]")
            print(f"  Difference: min_diff={m['min_diff']:.6f}, max_diff={m['max_diff']:.6f}")
            print()
    
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

