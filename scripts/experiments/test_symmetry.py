from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG
from g1_crawl.tasks.manager_based.g1_crawl.agents.symmetry_func import (
    mirror_joint_tensor,
    mirror_observation_policy,
    mirror_observation_critic
)

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_rotate_inverse

import torch
import numpy as np

# Add carb and omni for keyboard input handling
import carb
import omni


class SymmetryTestSceneCfg(InteractiveSceneCfg):
    """Scene configuration for symmetry testing with two robots."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Original robot (left) - keep fix_root_link=True for stable debugging
    RobotOriginal = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotOriginal",
        spawn=G1_CFG.spawn.replace(
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                fix_root_link=True
            )
        )
    )

    # Mirrored robot (right)
    RobotMirrored = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotMirrored",
        spawn=G1_CFG.spawn.replace(
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                fix_root_link=True
            )
        )
    )


def generate_random_joint_positions(robot, scale=1.0):
    """Generate random joint positions within joint limits.
    
    Args:
        robot: The robot asset
        scale: Scale factor for randomization (0.0 = default pose, 1.0 = full range)
    
    Returns:
        Tensor of random joint positions
    """
    joint_positions = robot.data.default_joint_pos.clone()
    
    # Get joint limits
    try:
        limits = robot.data.soft_joint_pos_limits[0]
    except:
        try:
            limits = robot.data.joint_pos_limits[0]
        except:
            limits = robot.data.hard_joint_pos_limits[0]
    
    lower = limits[:, 0]
    upper = limits[:, 1]
    
    # Generate random values within limits
    random_values = torch.rand(len(lower), device=robot.device)
    joint_positions[0] = lower + random_values * (upper - lower)
    
    # Blend with default pose based on scale
    joint_positions[0] = (1.0 - scale) * robot.data.default_joint_pos[0] + scale * joint_positions[0]
    
    return joint_positions


def apply_joint_positions(robot, joint_positions, position_offset):
    """Apply joint positions to a robot and position it in the scene.
    
    Args:
        robot: The robot asset
        joint_positions: Tensor of joint positions to apply
        position_offset: [x, y, z] offset for robot position in world
    """
    # Set robot position
    root_state = robot.data.default_root_state.clone()
    root_state[0, 0:3] += torch.tensor(position_offset, device=robot.device)
    
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.set_joint_position_target(joint_positions)
    robot.write_data_to_sim()


def print_joint_comparison(robot_original, robot_mirrored):
    """Print side-by-side comparison of original and mirrored joint values."""
    joint_names = robot_original.data.joint_names
    joint_pos_orig = robot_original.data.joint_pos[0]
    joint_pos_mirr = robot_mirrored.data.joint_pos[0]
    
    print("\n" + "=" * 100)
    print("JOINT COMPARISON: ORIGINAL vs MIRRORED")
    print("=" * 100)
    print(f"{'Index':<8} {'Joint Name':<30} {'Original':<12} {'Mirrored':<12} {'Diff':<12}")
    print("-" * 100)
    
    for i, name in enumerate(joint_names):
        orig = float(joint_pos_orig[i])
        mirr = float(joint_pos_mirr[i])
        diff = abs(orig - mirr)
        print(f"{i:<8} {name:<30} {orig:<12.4f} {mirr:<12.4f} {diff:<12.4f}")
    
    print("=" * 100)
    
    # Print expected symmetry relationships
    print("\nEXPECTED SYMMETRY BEHAVIOR:")
    print("  - Left/Right pairs should be swapped")
    print("  - Yaw/Roll joints should be inverted")
    print("  - Pitch joints should maintain sign")
    print("  - Non-paired joints (e.g., waist_yaw) should be inverted")
    print("=" * 100)


def construct_policy_observation(robot):
    """Construct policy observation similar to the actual environment.
    
    Policy observation structure (75 dims total):
    - projected_gravity (3)
    - velocity_commands (3) - will use dummy values
    - joint_pos (23)
    - joint_vel (23)
    - actions (23) - will use dummy values (last actions)
    """
    device = robot.device
    
    # Projected gravity (gravity in base frame)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device)
    projected_gravity = quat_rotate_inverse(robot.data.root_quat_w[0:1], gravity_w.unsqueeze(0))
    
    # Velocity commands (dummy values for testing)
    velocity_commands = torch.zeros(1, 3, device=device)
    
    # Joint positions and velocities
    joint_pos = robot.data.joint_pos[0:1]
    joint_vel = robot.data.joint_vel[0:1]
    
    # Actions (dummy - use current joint positions as proxy)
    actions = joint_pos.clone()
    
    # Concatenate
    obs = torch.cat([
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
        actions
    ], dim=-1)
    
    return obs


def construct_critic_observation(robot):
    """Construct critic observation similar to the actual environment.
    
    Critic observation structure (81 dims total):
    - base_lin_vel (3)
    - base_ang_vel (3)
    - projected_gravity (3)
    - velocity_commands (3) - will use dummy values
    - joint_pos (23)
    - joint_vel (23)
    - actions (23) - will use dummy values
    """
    device = robot.device
    
    # Base velocities
    base_lin_vel = robot.data.root_lin_vel_b[0:1]
    base_ang_vel = robot.data.root_ang_vel_b[0:1]
    
    # Projected gravity
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device)
    projected_gravity = quat_rotate_inverse(robot.data.root_quat_w[0:1], gravity_w.unsqueeze(0))
    
    # Velocity commands (dummy)
    velocity_commands = torch.zeros(1, 3, device=device)
    
    # Joint positions and velocities
    joint_pos = robot.data.joint_pos[0:1]
    joint_vel = robot.data.joint_vel[0:1]
    
    # Actions (dummy)
    actions = joint_pos.clone()
    
    # Concatenate
    obs = torch.cat([
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
        actions
    ], dim=-1)
    
    return obs


def print_observation_comparison(robot_original, robot_mirrored, obs_type="policy"):
    """Compare actual observations vs mirrored observations.
    
    The test: If robots are in mirrored poses, then:
    - mirror(obs_original) should ≈ obs_mirrored
    - mirror(obs_mirrored) should ≈ obs_original
    """
    # Construct observations from both robots
    if obs_type == "policy":
        obs_original = construct_policy_observation(robot_original)
        obs_mirrored = construct_policy_observation(robot_mirrored)
        mirror_func = mirror_observation_policy
        obs_labels = [
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "vel_cmd_x", "vel_cmd_y", "vel_cmd_z",
        ]
        # Add joint position labels
        joint_names = robot_original.data.joint_names
        obs_labels += [f"joint_pos_{name}" for name in joint_names]
        obs_labels += [f"joint_vel_{name}" for name in joint_names]
        obs_labels += [f"action_{name}" for name in joint_names]
    else:  # critic
        obs_original = construct_critic_observation(robot_original)
        obs_mirrored = construct_critic_observation(robot_mirrored)
        mirror_func = mirror_observation_critic
        obs_labels = [
            "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "vel_cmd_x", "vel_cmd_y", "vel_cmd_z",
        ]
        joint_names = robot_original.data.joint_names
        obs_labels += [f"joint_pos_{name}" for name in joint_names]
        obs_labels += [f"joint_vel_{name}" for name in joint_names]
        obs_labels += [f"action_{name}" for name in joint_names]
    
    # Apply mirroring functions
    # Note: mirror functions return stacked tensor [original, mirrored], so we take [1] for mirrored
    mirrored_from_original = mirror_func(obs_original)
    if mirrored_from_original is not None:
        mirrored_from_original = mirrored_from_original[1:2]  # Take the mirrored version
    
    mirrored_from_mirrored = mirror_func(obs_mirrored)
    if mirrored_from_mirrored is not None:
        mirrored_from_mirrored = mirrored_from_mirrored[1:2]  # Take the mirrored version
    
    print("\n" + "=" * 120)
    print(f"OBSERVATION COMPARISON ({obs_type.upper()})")
    print("=" * 120)
    print("Testing: mirror(obs_original) should ≈ obs_mirrored (if poses are truly mirrored)")
    print("=" * 120)
    print(f"{'Index':<6} {'Observation':<35} {'Obs_Orig':<12} {'Mirror(Orig)':<12} {'Obs_Mirr':<12} {'Error':<12}")
    print("-" * 120)
    
    obs_orig_flat = obs_original[0]
    obs_mirr_flat = obs_mirrored[0]
    mirrored_orig_flat = mirrored_from_original[0] if mirrored_from_original is not None else torch.zeros_like(obs_orig_flat)
    
    total_error = 0.0
    max_error = 0.0
    max_error_idx = 0
    
    for i in range(len(obs_orig_flat)):
        obs_o = float(obs_orig_flat[i])
        mirr_o = float(mirrored_orig_flat[i])
        obs_m = float(obs_mirr_flat[i])
        error = abs(mirr_o - obs_m)
        
        total_error += error
        if error > max_error:
            max_error = error
            max_error_idx = i
        
        label = obs_labels[i] if i < len(obs_labels) else f"obs_{i}"
        print(f"{i:<6} {label:<35} {obs_o:<12.4f} {mirr_o:<12.4f} {obs_m:<12.4f} {error:<12.6f}")
    
    print("-" * 120)
    print(f"Total L1 Error: {total_error:.6f}")
    print(f"Max Error: {max_error:.6f} at index {max_error_idx} ({obs_labels[max_error_idx] if max_error_idx < len(obs_labels) else 'unknown'})")
    print("=" * 120)
    
    # Also test the reverse
    print("\nREVERSE TEST: mirror(obs_mirrored) should ≈ obs_original")
    print("-" * 120)
    
    mirrored_mirr_flat = mirrored_from_mirrored[0] if mirrored_from_mirrored is not None else torch.zeros_like(obs_mirr_flat)
    
    reverse_total_error = 0.0
    for i in range(len(obs_mirr_flat)):
        obs_m = float(obs_mirr_flat[i])
        mirr_m = float(mirrored_mirr_flat[i])
        obs_o = float(obs_orig_flat[i])
        error = abs(mirr_m - obs_o)
        reverse_total_error += error
    
    print(f"Reverse Total L1 Error: {reverse_total_error:.6f}")
    print("=" * 120)
    
    # Summary
    if total_error < 0.01 and reverse_total_error < 0.01:
        print("✓ OBSERVATION SYMMETRY TEST PASSED!")
    else:
        print("✗ OBSERVATION SYMMETRY TEST FAILED - Check errors above")
    print("=" * 120)


def run_symmetry_test(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the symmetry test visualization with keyboard controls."""
    robot_original = scene["RobotOriginal"]
    robot_mirrored = scene["RobotMirrored"]
    
    # Position robots side by side (2 meters apart in Y, 1 meter above ground)
    robot_original_offset = [0.0, -1.0, 1.0]
    robot_mirrored_offset = [0.0, 1.0, 1.0]
    
    # Current randomization scale
    random_scale = 0.5
    
    # Generate initial random pose
    original_joint_pos = generate_random_joint_positions(robot_original, scale=random_scale)
    
    # Mirror the joint positions
    mirrored_joint_pos = torch.zeros_like(original_joint_pos)
    mirror_joint_tensor(original_joint_pos, mirrored_joint_pos, offset=0)
    
    # Apply to robots
    apply_joint_positions(robot_original, original_joint_pos, robot_original_offset)
    apply_joint_positions(robot_mirrored, mirrored_joint_pos, robot_mirrored_offset)
    scene.write_data_to_sim()
    
    print("\n[INFO] Initial random pose applied")
    print(f"[INFO] Randomization scale: {random_scale:.2f}")
    
    # Set up keyboard input handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {
        "R": False,
        "D": False,
        "I": False,
        "O": False,
        "C": False,
        "UP": False,
        "DOWN": False,
        "SPACE": False,
        "ESCAPE": False,
    }
    
    def on_keyboard_event(event):
        nonlocal random_scale, original_joint_pos, mirrored_joint_pos
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                print("\n[DEBUG] Resetting to default pose...")
                
                # Reset both robots to default
                default_pos = robot_original.data.default_joint_pos.clone()
                mirrored_default = torch.zeros_like(default_pos)
                mirror_joint_tensor(default_pos, mirrored_default, offset=0)
                
                apply_joint_positions(robot_original, default_pos, robot_original_offset)
                apply_joint_positions(robot_mirrored, mirrored_default, robot_mirrored_offset)
                scene.write_data_to_sim()
                
                original_joint_pos = default_pos
                mirrored_joint_pos = mirrored_default
                print("[DEBUG] Reset complete")
                
            elif event.input.name == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                print(f"\n[DEBUG] Generating new random pose (scale={random_scale:.2f})...")
                
                # Generate new random pose
                original_joint_pos = generate_random_joint_positions(robot_original, scale=random_scale)
                mirrored_joint_pos = torch.zeros_like(original_joint_pos)
                mirror_joint_tensor(original_joint_pos, mirrored_joint_pos, offset=0)
                
                # Apply to robots
                apply_joint_positions(robot_original, original_joint_pos, robot_original_offset)
                apply_joint_positions(robot_mirrored, mirrored_joint_pos, robot_mirrored_offset)
                scene.write_data_to_sim()
                
                print("[DEBUG] New random pose applied")
                
            elif event.input.name == "I" and not keys_pressed["I"]:
                keys_pressed["I"] = True
                print("\n[DEBUG] Printing joint comparison...")
                # Small delay to ensure sim updates
                sim.step()
                scene.update(sim.get_physics_dt())
                print_joint_comparison(robot_original, robot_mirrored)
                
            elif event.input.name == "O" and not keys_pressed["O"]:
                keys_pressed["O"] = True
                print("\n[DEBUG] Testing POLICY observation symmetry...")
                # Small delay to ensure sim updates
                sim.step()
                scene.update(sim.get_physics_dt())
                print_observation_comparison(robot_original, robot_mirrored, obs_type="policy")
                
            elif event.input.name == "C" and not keys_pressed["C"]:
                keys_pressed["C"] = True
                print("\n[DEBUG] Testing CRITIC observation symmetry...")
                # Small delay to ensure sim updates
                sim.step()
                scene.update(sim.get_physics_dt())
                print_observation_comparison(robot_original, robot_mirrored, obs_type="critic")
                
            elif event.input.name == "UP" and not keys_pressed["UP"]:
                keys_pressed["UP"] = True
                random_scale = min(1.0, random_scale + 0.1)
                print(f"\n[DEBUG] Increased randomization scale to {random_scale:.2f}")
                
            elif event.input.name == "DOWN" and not keys_pressed["DOWN"]:
                keys_pressed["DOWN"] = True
                random_scale = max(0.0, random_scale - 0.1)
                print(f"\n[DEBUG] Decreased randomization scale to {random_scale:.2f}")
                
            elif event.input.name == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                print("\n[DEBUG] Swapping robots (testing bidirectional symmetry)...")
                
                # Swap the poses
                original_joint_pos, mirrored_joint_pos = mirrored_joint_pos, original_joint_pos
                
                apply_joint_positions(robot_original, original_joint_pos, robot_original_offset)
                apply_joint_positions(robot_mirrored, mirrored_joint_pos, robot_mirrored_offset)
                scene.write_data_to_sim()
                
                print("[DEBUG] Robots swapped")
                
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    # Subscribe to keyboard events
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print(f"\n" + "=" * 80)
    print("G1 SYMMETRY FUNCTION TEST")
    print("=" * 80)
    print("Two robots displayed side-by-side:")
    print("  LEFT  - Original pose")
    print("  RIGHT - Mirrored pose")
    print("\nControls:")
    print("  R        - Reset to default pose")
    print("  D        - Generate new random pose")
    print("  I        - Print joint comparison (original vs mirrored)")
    print("  O        - Test POLICY observation symmetry")
    print("  C        - Test CRITIC observation symmetry")
    print("  UP       - Increase randomization scale")
    print("  DOWN     - Decrease randomization scale")
    print("  SPACE    - Swap robots (test bidirectional symmetry)")
    print("  ESC      - Exit")
    print("=" * 80)
    print("\nOBSERVATION TESTING:")
    print("  The observation tests verify that mirror(obs_original) ≈ obs_mirrored")
    print("  This validates that the symmetry functions correctly transform observations")
    print("  from one side to match what the mirrored robot actually observes.")
    print("=" * 80)
    
    sim_dt = sim.get_physics_dt()
    
    try:
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("Exiting symmetry test...")
                break
                
            sim.step()
            scene.update(sim_dt)
    
    finally:
        # Clean up keyboard subscription
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 2.5], [0.0, 0.0, 1.0])
    
    # Create scene with two robots
    scene_cfg = SymmetryTestSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    print("[INFO]: G1 Symmetry Function Test starting...")
    
    # Run the symmetry test
    run_symmetry_test(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

