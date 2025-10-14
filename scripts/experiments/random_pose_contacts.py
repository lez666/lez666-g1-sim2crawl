from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

import time
import torch

# Optional keyboard input handling
import carb
import omni


def create_scene_cfg():
    """Create a simple scene with ground, light, robot, and contact sensors."""

    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_CFG.spawn.replace(
                rigid_props=G1_CFG.spawn.rigid_props.replace(disable_gravity=False),
                articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=False),
            ),
        )
        
        # Add contact sensor to track all body contacts with red sphere visualization
        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=2,
            debug_vis=True,
            visualizer_cfg=VisualizationMarkersCfg(
                prim_path="/Visuals/ContactSensor",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            ),
        )

    return SimpleSceneCfg


def build_joint_name_to_index_map(asset) -> dict:
    joint_names = [str(n) for n in asset.data.joint_names]
    name_to_index = {name: int(i) for i, name in enumerate(joint_names)}
    return name_to_index


def generate_random_pose(scene: InteractiveScene, name_to_index: dict) -> torch.Tensor:
    """Generate and apply a random pose to the robot. Returns the target joint positions."""
    device = scene["Robot"].device
    robot = scene["Robot"]
    
    # Random root position - spawn above ground
    origin = scene.env_origins.to(device=device)[0] if hasattr(scene, "env_origins") else torch.zeros(3, device=device)
    new_root_state = torch.zeros(1, 13, device=device)
    
    # Position: x=0, y=0, z between 0.5 and 1.5 meters (spawn in air to drop)
    base_z = torch.rand(1, device=device) * 1.0 + 0.5
    base_pos_tensor = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    base_pos_tensor[2] = base_z
    
    # Random orientation (quaternion)
    # Generate random roll, pitch, yaw and convert to quaternion
    roll = (torch.rand(1, device=device) - 0.5) * 0.6  # +/- 0.3 rad
    pitch = (torch.rand(1, device=device) - 0.5) * 0.6
    yaw = (torch.rand(1, device=device) - 0.5) * 2.0  # +/- 1.0 rad
    
    # Simple quaternion from RPY
    half_r = 0.5 * roll
    half_p = 0.5 * pitch
    half_y = 0.5 * yaw
    
    cr = torch.cos(half_r)
    sr = torch.sin(half_r)
    cp = torch.cos(half_p)
    sp = torch.sin(half_p)
    cy = torch.cos(half_y)
    sy = torch.sin(half_y)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z], dim=1).to(dtype=torch.float32)
    quat = quat / torch.linalg.norm(quat, dim=1, keepdim=True)
    
    new_root_state[0, :3] = base_pos_tensor + origin
    new_root_state[0, 3:7] = quat[0]
    new_root_state[0, 7:13] = 0.0
    
    robot.write_root_pose_to_sim(new_root_state[:, :7])
    robot.write_root_velocity_to_sim(new_root_state[:, 7:])
    
    # Random joint positions within safe limits
    joint_positions = robot.data.default_joint_pos.clone().to(device=device)
    joint_velocities = torch.zeros_like(joint_positions)
    
    # Randomize each joint within a fraction of its limits
    joint_pos_limits = robot.data.soft_joint_pos_limits
    for joint_name, j_idx in name_to_index.items():
        if joint_pos_limits is not None and j_idx < joint_pos_limits.shape[1]:
            lower = joint_pos_limits[0, j_idx, 0]
            upper = joint_pos_limits[0, j_idx, 1]
            # Random value within 20-80% of range to stay safe
            range_size = upper - lower
            random_val = lower + torch.rand(1, device=device) * range_size * 0.6 + range_size * 0.2
            joint_positions[0, j_idx] = random_val
        else:
            # No limits available, use small random offset from default
            random_offset = (torch.rand(1, device=device) - 0.5) * 0.5
            joint_positions[0, j_idx] += random_offset
    
    # Set initial joint state
    robot.write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()
    
    return joint_positions


def print_contact_info(scene: InteractiveScene) -> None:
    """Print detailed contact information from the contact sensor."""
    contact_sensor: ContactSensor = scene.sensors["contact_sensor"]
    
    # Update sensor data
    contact_sensor.update(dt=0.0)
    
    # Get contact data
    net_forces = contact_sensor.data.net_forces_w  # Current net forces in world frame
    net_forces_history = contact_sensor.data.net_forces_w_history  # Historical forces
    force_matrix = contact_sensor.data.force_matrix_w  # Detailed force matrix
    
    print("\n" + "="*80)
    print("CONTACT INFORMATION")
    print("="*80)
    
    # Print basic info
    print(f"Number of bodies tracked: {net_forces.shape[1]}")
    print(f"Net forces shape: {net_forces.shape}")  # (num_envs, num_bodies, 3)
    
    # Get body names from the contact sensor
    body_names = []
    if hasattr(contact_sensor, 'body_names'):
        body_names = contact_sensor.body_names
    elif hasattr(contact_sensor, '_body_physx_view') and hasattr(contact_sensor._body_physx_view, 'body_names'):
        body_names = contact_sensor._body_physx_view.body_names
    else:
        # Fallback: get body names from the robot asset
        robot = scene["Robot"]
        if hasattr(robot.data, 'body_names'):
            body_names = robot.data.body_names
    
    # Compute force magnitudes
    force_magnitudes = torch.norm(net_forces[0], dim=-1)  # (num_bodies,)
    
    # Find bodies with non-zero contact
    contact_threshold = 0.1  # Newtons
    bodies_in_contact = force_magnitudes > contact_threshold
    
    print(f"\nBodies in contact (threshold: {contact_threshold}N):")
    if torch.any(bodies_in_contact):
        for body_idx in range(len(force_magnitudes)):
            if bodies_in_contact[body_idx]:
                force_mag = force_magnitudes[body_idx].item()
                force_vec = net_forces[0, body_idx]
                body_name = body_names[body_idx] if body_idx < len(body_names) else f"Body_{body_idx}"
                print(f"  {body_name}: {force_mag:.2f}N - Force: [{force_vec[0]:.2f}, {force_vec[1]:.2f}, {force_vec[2]:.2f}]")
    else:
        print("  No contacts detected")
    
    # Print summary statistics
    total_contact_force = torch.sum(force_magnitudes).item()
    max_contact_force = torch.max(force_magnitudes).item()
    num_contacts = torch.sum(bodies_in_contact).item()
    
    print(f"\nSummary:")
    print(f"  Total contact force: {total_contact_force:.2f}N")
    print(f"  Maximum contact force: {max_contact_force:.2f}N")
    print(f"  Number of bodies in contact: {int(num_contacts)}")
    
    # Check historical forces for any recent contacts
    if net_forces_history is not None:
        historical_magnitudes = torch.norm(net_forces_history[0], dim=-1)  # (history_length, num_bodies)
        max_historical = torch.max(historical_magnitudes, dim=0)[0]  # Max over history for each body
        print(f"\nMaximum historical forces (across last {net_forces_history.shape[1]} steps):")
        print(f"  Max force in history: {torch.max(max_historical).item():.2f}N")
    
    print("="*80 + "\n")


def run_random_pose_viewer(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    name_to_index = build_joint_name_to_index_map(scene["Robot"])
    robot = scene["Robot"]
    
    # Target pose that robot will try to hold
    target_joint_positions = None
    
    # Keyboard setup
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {"R": False, "ESCAPE": False, "SPACE": False}
    auto_mode = False
    
    def on_keyboard_event(event):
        nonlocal keys_pressed, auto_mode, target_joint_positions
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                print("\n[Generating new random pose...]")
                target_joint_positions = generate_random_pose(scene, name_to_index)
            elif key == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                auto_mode = not auto_mode
                print(f"\nAuto-mode: {'ON' if auto_mode else 'OFF'}")
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    # Generate initial random pose
    print("\n[Generating initial random pose...]")
    target_joint_positions = generate_random_pose(scene, name_to_index)
    
    # Let physics settle
    print("[Settling physics...]")
    for _ in range(100):
        # Command robot to hold the target pose
        robot.set_joint_position_target(target_joint_positions)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    
    print("\n" + "="*80)
    print("Random Pose Contact Viewer")
    print("="*80)
    print("\nControls:")
    print("  R - Generate new random pose and drop")
    print("  SPACE - Toggle auto-mode (new pose every 5 seconds)")
    print("  ESC - Exit")
    print("\nThe robot will hold each random pose as it falls to the ground.")
    print("Contact points are visualized with markers in the scene.")
    print("="*80)
    
    # Print initial contact info
    print_contact_info(scene)
    
    last_auto_time = time.time()
    auto_interval = 5.0  # seconds
    
    try:
        sim_dt = sim.get_physics_dt()
        step_count = 0
        settle_steps = 200  # Steps to let physics settle after pose change (longer to see impact)
        settling = False
        settle_counter = 0
        
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("\n[Exiting...]")
                break
            
            # Auto-mode: generate new pose periodically
            current_time = time.time()
            if auto_mode and (current_time - last_auto_time >= auto_interval):
                print(f"\n[Auto-mode: Generating new random pose...]")
                target_joint_positions = generate_random_pose(scene, name_to_index)
                last_auto_time = current_time
                settling = True
                settle_counter = 0
            
            # Manual pose generation
            if keys_pressed["R"]:
                settling = True
                settle_counter = 0
            
            # Command robot to actively hold the target pose
            if target_joint_positions is not None:
                robot.set_joint_position_target(target_joint_positions)
            
            # Settle physics after pose change
            if settling:
                settle_counter += 1
                if settle_counter >= settle_steps:
                    settling = False
                    settle_counter = 0
                    # Print contact info after settling
                    print_contact_info(scene)
            
            robot.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step_count += 1
            
    finally:
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda", dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Scene
    scene_cfg_class = create_scene_cfg()
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Start simulation
    sim.reset()
    print("[INFO]: Setup complete...")
    
    run_random_pose_viewer(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

