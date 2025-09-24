import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
import threading
import tkinter as tk
from dataclasses import dataclass
from typing import Union
from utils import (
    default_angles_config,
    crawl_angles,
    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)


# SCENE_XML_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/g1_description/scene_torso_collision_test.xml")
SCENE_XML_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/g1_description/scene_mjx_alt.xml")

START_FACE_DOWN: bool = True
DROP_HEIGHT_M: float = 0.5

# NN control constants (simple, no args/params)
NN_SWITCH_SEC: float = 1
NN_POLICY_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/policy.pt")
NN_ACTION_SCALE: float = 0.5
NN_N_SUBSTEPS: int = 4

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


class TorchController:
    """Minimal PyTorch controller: fixed commands, no keyboard."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        state: Union["ControlState", None] = None,
    ) -> None:
        self._policy = torch.load(policy_path, weights_only=False)
        self._policy.eval()

        self._default_angles = default_angles.astype(np.float32)
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._action_scale = float(action_scale)
        print("ACTION_SCALE", self._action_scale)
        self._n_substeps = int(n_substeps)
        self._counter = 0
        self._state = state

        # Initialize joint mappings once
        init_joint_mappings()

    def _get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        # Project gravity into IMU frame
        world_gravity = model.opt.gravity
        world_gravity = world_gravity / np.linalg.norm(world_gravity)
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        projected_gravity = imu_xmat.T @ world_gravity

        # Commanded velocities [lin_vel_z, lin_vel_y, ang_vel_x]
        state = getattr(self, "_state", None)
        if state is not None:
            lin_vel_z = float(state.lin_vel_z)
            ang_vel_x = float(state.ang_vel_x)
        else:
            lin_vel_z = 1.5
            ang_vel_x = 0.0
        velocity_commands = np.array([lin_vel_z, 0.0, ang_vel_x], dtype=np.float32)

        # Joint states (MuJoCo order) → PyTorch order
        joint_pos_mujoco = data.qpos[7:] - self._default_angles
        joint_vel_mujoco = data.qvel[6:]
        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        actions_pytorch = remap_mujoco_to_pytorch(self._last_action)

        obs = np.hstack([
            projected_gravity,
            velocity_commands,
            joint_pos_pytorch,
            joint_vel_pytorch,
            actions_pytorch,
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        obs = self._get_obs(model, data)
        obs_tensor = torch.from_numpy(obs).float()
        with torch.no_grad():
            action_tensor = self._policy(obs_tensor)
            pytorch_pred = action_tensor.numpy()

        mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)
        self._last_action = mujoco_pred.copy()
        data.ctrl[:] = mujoco_pred * self._action_scale + self._default_angles
        # data.ctrl[:] = self._default_angles



@dataclass
class ControlState:
    mode: str = "hold"  # "hold" or "nn"
    requested_reset: bool = False
    action_scale: float = NN_ACTION_SCALE
    n_substeps: int = NN_N_SUBSTEPS
    lin_vel_z: float = 1.5  # Forward velocity
    ang_vel_x: float = 0.0  # Angular velocity around x-axis
    quit: bool = False


class SimpleUI:
    def __init__(self, state: ControlState) -> None:
        self.state = state
        self._root: tk.Tk | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("G1 Crawl Controls")

        frm = tk.Frame(self._root)
        frm.pack(padx=8, pady=8)

        # Row 0: Mode selection
        tk.Button(frm, text="Hold Pose", width=12, command=lambda: self._set_mode("hold")).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(frm, text="NN Control", width=12, command=lambda: self._set_mode("nn")).grid(row=0, column=1, padx=4, pady=4)

        # Row 1: Reset / Quit
        tk.Button(frm, text="Reset Drop", width=12, command=self._reset).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(frm, text="Quit", width=12, command=self._quit).grid(row=1, column=1, padx=4, pady=4)

        # Row 2: Action scale slider
        tk.Label(frm, text="Action Scale").grid(row=2, column=0, sticky="e", padx=4)
        action_scale_var = tk.DoubleVar(value=self.state.action_scale)

        def on_action_scale(val: str) -> None:
            try:
                self.state.action_scale = float(val)
            except Exception:
                pass

        tk.Scale(
            frm,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=action_scale_var,
            command=on_action_scale,
            length=220,
        ).grid(row=2, column=1, sticky="w", padx=4)

        # Row 3: Substeps slider
        tk.Label(frm, text="Substeps").grid(row=3, column=0, sticky="e", padx=4)
        n_substeps_var = tk.IntVar(value=self.state.n_substeps)

        def on_substeps(val: str) -> None:
            try:
                self.state.n_substeps = max(1, int(float(val)))
            except Exception:
                pass

        tk.Scale(
            frm,
            from_=1,
            to=8,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=n_substeps_var,
            command=on_substeps,
            length=220,
        ).grid(row=3, column=1, sticky="w", padx=4)

        # Row 4: Forward velocity slider
        tk.Label(frm, text="Forward Vel").grid(row=4, column=0, sticky="e", padx=4)
        lin_vel_z_var = tk.DoubleVar(value=self.state.lin_vel_z)

        def on_lin_vel_z(val: str) -> None:
            try:
                self.state.lin_vel_z = float(val)
            except Exception:
                pass

        tk.Scale(
            frm,
            from_=0,
            to=1.5,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=lin_vel_z_var,
            command=on_lin_vel_z,
            length=220,
        ).grid(row=4, column=1, sticky="w", padx=4)

        # Row 5: Angular velocity slider
        tk.Label(frm, text="Angular Vel").grid(row=5, column=0, sticky="e", padx=4)
        ang_vel_x_var = tk.DoubleVar(value=self.state.ang_vel_x)

        def on_ang_vel_x(val: str) -> None:
            try:
                self.state.ang_vel_x = float(val)
            except Exception:
                pass

        tk.Scale(
            frm,
            from_=-1.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=ang_vel_x_var,
            command=on_ang_vel_x,
            length=220,
        ).grid(row=5, column=1, sticky="w", padx=4)

        self._root.protocol("WM_DELETE_WINDOW", self._quit)
        self._root.mainloop()

    def _set_mode(self, mode: str) -> None:
        self.state.mode = str(mode)

    def _reset(self) -> None:
        self.state.requested_reset = True

    def _quit(self) -> None:
        self.state.quit = True
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    # Find a FREE joint if present
    free_qpos_addr: int | None = None
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:
            free_qpos_addr = int(model.jnt_qposadr[j])
            break

    # Start face-down with a small drop if a FREE joint is present
    if START_FACE_DOWN and free_qpos_addr is not None:
        data.qpos[free_qpos_addr + 0] = 0.0
        data.qpos[free_qpos_addr + 1] = 0.0
        data.qpos[free_qpos_addr + 2] = float(max(0.0, DROP_HEIGHT_M))
        qw, qx, qy, qz = rpy_to_quat(0.0, math.pi / 2.0, 0.0)
        data.qpos[free_qpos_addr + 3] = qw
        data.qpos[free_qpos_addr + 4] = qx
        data.qpos[free_qpos_addr + 5] = qy
        data.qpos[free_qpos_addr + 6] = qz

    # Apply crawl_angles to all hinge/slide joints before first forward
    k = 0
    for j in range(model.njnt):
        if model.jnt_type[j] in (2, 3):  # hinge or slide
            if k < len(crawl_angles):
                adr = model.jnt_qposadr[j]
                data.qpos[adr] = float(crawl_angles[k])
                k += 1

    mujoco.mj_forward(model, data)

    # Build mapping from joint id -> actuator id (names are aligned in XML)
    joint_to_actuator: dict[int, int] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] not in (2, 3):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname is None:
            continue
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        except Exception:
            aid = -1
        if aid != -1:
            joint_to_actuator[int(j)] = int(aid)

    controlled_joint_ids = sorted(joint_to_actuator.keys())

    # Hold targets come directly from crawl_angles (MuJoCo hinge/slide joint order)
    # Map by XML joint order (ids increase with definition order)
    hold_targets: dict[int, float] = {}
    for k, j in enumerate(controlled_joint_ids):
        if k < len(crawl_angles):
            hold_targets[j] = float(crawl_angles[k])
        else:
            hold_targets[j] = 0.0

    # Start simple UI in background
    ctrl_state = ControlState(action_scale=NN_ACTION_SCALE, n_substeps=NN_N_SUBSTEPS)
    ui = SimpleUI(state=ctrl_state)

    # Prepare NN controller and switching timer
    controller = TorchController(
        policy_path=NN_POLICY_PATH.as_posix(),
        default_angles=np.array(default_angles_config, dtype=np.float32),
        n_substeps=NN_N_SUBSTEPS,
        action_scale=NN_ACTION_SCALE,
        state=ctrl_state,
    )
    t0 = time.time()

    with viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            if ctrl_state.quit:
                break
            elapsed = time.time() - t0
            # Process UI intents
            if ctrl_state.requested_reset:
                # Re-apply face-down drop (if free joint)
                if free_qpos_addr is not None:
                    data.qpos[free_qpos_addr + 0] = 0.0
                    data.qpos[free_qpos_addr + 1] = 0.0
                    data.qpos[free_qpos_addr + 2] = float(max(0.0, DROP_HEIGHT_M))
                    qw, qx, qy, qz = rpy_to_quat(0.0, math.pi / 2.0, 0.0)
                    data.qpos[free_qpos_addr + 3] = qw
                    data.qpos[free_qpos_addr + 4] = qx
                    data.qpos[free_qpos_addr + 5] = qy
                    data.qpos[free_qpos_addr + 6] = qz
                # Re-apply crawl_angles to hinge/slide joints
                k = 0
                for j in range(model.njnt):
                    if model.jnt_type[j] in (2, 3):
                        if k < len(crawl_angles):
                            adr = model.jnt_qposadr[j]
                            data.qpos[adr] = float(crawl_angles[k])
                            k += 1
                mujoco.mj_forward(model, data)
                ctrl_state.requested_reset = False

            # Sync controller params from UI
            controller._action_scale = float(ctrl_state.action_scale)
            controller._n_substeps = int(ctrl_state.n_substeps)

            # Control mode: initial hold for NN_SWITCH_SEC unless user forces NN
            if elapsed < NN_SWITCH_SEC and ctrl_state.mode != "nn":
                mode = "hold"
            else:
                mode = ctrl_state.mode

            if mode == "hold":
                for j in controlled_joint_ids:
                    aid = joint_to_actuator[j]
                    data.ctrl[aid] = hold_targets[j]
            else:
                controller.get_control(model, data)

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep * 0.5))


if __name__ == "__main__":
    main()


