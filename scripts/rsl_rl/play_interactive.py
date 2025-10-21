# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
import isaaclab.utils.math as math_utils

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import g1_crawl.tasks  # noqa: F401


try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    _DEBUG_DRAW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    omni_debug_draw = None
    _DEBUG_DRAW_AVAILABLE = False


def _get_robot_pose(env):
    try:
        base_env = env.unwrapped
    except Exception:
        base_env = env
    try:
        robot = base_env.scene["robot"]
        pos_w = robot.data.root_pos_w[0]
        quat_w = robot.data.root_quat_w[0]
        return pos_w, quat_w
    except Exception:
        return None, None


def _check_joint_limits(env, print_interval_steps: int = 500):
    """Check if any joints exceed their limits and print warnings.
    
    Args:
        env: The environment (will unwrap to get robot)
        print_interval_steps: Only print violations every N steps to avoid spam
    """
    try:
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        
        # Get joint limits (prefer soft limits, fall back to hard limits)
        limits = None
        limits_label = None
        try:
            limits = robot.data.soft_joint_pos_limits[0]
            limits_label = "soft"
        except Exception:
            try:
                limits = robot.data.joint_pos_limits[0]
                limits_label = "hard"
            except Exception:
                return  # No limits available
        
        if limits is None:
            return
        
        # Get current joint positions
        joint_pos = robot.data.joint_pos[0]
        joint_names = robot.data.joint_names
        
        # Check for violations
        lower = limits[:, 0]
        upper = limits[:, 1]
        
        below_mask = joint_pos < lower
        above_mask = joint_pos > upper
        
        violations = []
        for i in range(len(joint_names)):
            if below_mask[i]:
                violations.append((
                    joint_names[i],
                    float(joint_pos[i]),
                    float(lower[i]),
                    float(upper[i]),
                    "below"
                ))
            elif above_mask[i]:
                violations.append((
                    joint_names[i],
                    float(joint_pos[i]),
                    float(lower[i]),
                    float(upper[i]),
                    "above"
                ))
        
        if violations:
            print("\n" + "=" * 80)
            print(f"⚠️  JOINT LIMIT VIOLATIONS ({limits_label} limits):")
            print("=" * 80)
            for name, pos, lo, hi, direction in violations:
                exceed = pos - hi if direction == "above" else lo - pos
                print(f"  {name}: {pos:.4f} {'>' if direction == 'above' else '<'} [{lo:.4f}, {hi:.4f}] (exceeds by {exceed:.4f} rad)")
            print("=" * 80 + "\n")
    
    except Exception as e:
        # Silently ignore if robot structure doesn't match expectations
        pass


def _viz_keyboard_command_relative_to_base(env, cmd_vals, duration: float = 0.1):
    if not _DEBUG_DRAW_AVAILABLE:
        return
    try:
        vx = float(cmd_vals[0])
        vy = float(cmd_vals[1])
        wz = float(cmd_vals[2])
    except Exception:
        return
    pos_w, quat_w = _get_robot_pose(env)
    if pos_w is None or quat_w is None:
        return
    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    base_pos = pos_w.detach().cpu().numpy()
    anchor = np.array([float(base_pos[0]), float(base_pos[1]), float(base_pos[2] + 0.6)], dtype=float)
    arrow_scale = 1.5
    line_start_points = []
    line_end_points = []
    line_colors = []
    line_sizes = []
    # Linear velocity arrow: map teleop (v_x forward, v_y lateral) to base (z, y), then rotate to world and project to XY
    vec_b = torch.tensor([0.0, vy, vx], device=quat_w.device, dtype=quat_w.dtype).unsqueeze(0)
    vec_w = math_utils.quat_apply(quat_w.unsqueeze(0), vec_b).squeeze(0).detach().cpu().numpy()
    vec_w[2] = 0.0
    end = anchor + vec_w * arrow_scale
    line_start_points.append([anchor[0], anchor[1], anchor[2]])
    line_end_points.append([end[0], end[1], end[2]])
    line_colors.append((0.1, 0.9, 0.2, 1.0))
    line_sizes.append(4.0)
    # Arrowhead in XY plane
    dir2d = np.array([end[0] - anchor[0], end[1] - anchor[1]], dtype=float)
    n2 = np.linalg.norm(dir2d)
    if n2 > 1e-6:
        dir2d = dir2d / n2
        ang = np.radians(25.0)
        ca, sa = np.cos(ang), np.sin(ang)
        head_len = 0.3
        h1 = np.array([-dir2d[0] * ca + dir2d[1] * sa, -dir2d[1] * ca - dir2d[0] * sa])
        h2 = np.array([-dir2d[0] * ca - dir2d[1] * sa, -dir2d[1] * ca + dir2d[0] * sa])
        head1 = [end[0] + h1[0] * head_len, end[1] + h1[1] * head_len, end[2]]
        head2 = [end[0] + h2[0] * head_len, end[1] + h2[1] * head_len, end[2]]
        line_start_points.append([end[0], end[1], end[2]])
        line_end_points.append(head1)
        line_colors.append((0.1, 0.9, 0.2, 1.0))
        line_sizes.append(3.0)
        line_start_points.append([end[0], end[1], end[2]])
        line_end_points.append(head2)
        line_colors.append((0.1, 0.9, 0.2, 1.0))
        line_sizes.append(3.0)
    # Yaw rate arrow: show a curved sense using XY arrow perpendicular to forward for sign clarity
    # Build a small sideways vector in world aligned with base +Y
    y_b = torch.tensor([0.0, 1.0, 0.0], device=quat_w.device, dtype=quat_w.dtype).unsqueeze(0)
    y_w = math_utils.quat_apply(quat_w.unsqueeze(0), y_b).squeeze(0).detach().cpu().numpy()
    y_w[2] = 0.0
    n = np.linalg.norm(y_w[:2])
    if n > 1e-6:
        y_w[:2] = y_w[:2] / n
    yaw_scale = 0.8
    yaw_len = max(min(abs(wz) * yaw_scale, 1.5), 0.0)
    yaw_dir = 1.0 if wz >= 0.0 else -1.0
    off = 0.5
    y_start = anchor + y_w * off
    y_end = y_start + y_w * (yaw_dir * yaw_len)
    line_start_points.append([y_start[0], y_start[1], y_start[2]])
    line_end_points.append([y_end[0], y_end[1], y_end[2]])
    line_colors.append((0.1, 0.9, 1.0, 1.0))
    line_sizes.append(4.0)
    draw.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    config = Se2KeyboardCfg(
        v_x_sensitivity=1.2,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=.9,
    )
    teleop_interface = Se2Keyboard(config)
    teleop_interface.reset()


    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    # Joint limit monitoring (print every N steps to avoid spam)
    check_limits_interval = 500
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        command = teleop_interface.advance()
        # print(command)
        _viz_keyboard_command_relative_to_base(env, command)

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # command = [0.99, 0.0, 0.0]  
            obs[:, 3:6] = torch.tensor(command, device=obs.device, dtype=obs.dtype)
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # Check joint limits periodically
        if timestep % check_limits_interval == 0:
            _check_joint_limits(env, check_limits_interval)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
