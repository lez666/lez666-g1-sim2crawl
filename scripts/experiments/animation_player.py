from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

import json
from typing import Dict, List, Tuple
import time

import torch

import carb
import omni
import omni.ui as ui


def _assert_keys(data: dict, required: List[str], where: str) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys {missing} in {where}")


def create_scene_cfg():
    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_CFG.spawn.replace(
                rigid_props=G1_CFG.spawn.rigid_props.replace(disable_gravity=True),
                articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=False),
            ),
        )

    return SimpleSceneCfg


def load_animation_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _assert_keys(data, ["frames", "metadata"], "animation json root")
    _assert_keys(data["metadata"], ["base", "joints", "qpos_labels"], "metadata")
    _assert_keys(data["metadata"]["base"], ["pos_indices", "quat_indices"], "metadata.base")
    return data


def build_joint_name_to_index_map(asset) -> Dict[str, int]:
    joint_names = [str(n) for n in asset.data.joint_names]
    mapping = {name: int(i) for i, name in enumerate(joint_names)}
    if len(mapping) == 0:
        raise RuntimeError("Robot asset reports zero joints; cannot play animation")
    return mapping


def build_animation_index_mapping(metadata: Dict, name_to_index: Dict[str, int]) -> Tuple[List[int], List[int]]:
    # Returns (anim_joint_indices, robot_joint_indices) aligned lists
    anim_joint_indices: List[int] = []
    robot_joint_indices: List[int] = []

    joints_meta = metadata["joints"]
    for j in joints_meta:
        name = j["name"]
        qposadr = int(j["qposadr"])  # starting index of this joint in qpos
        if name == "floating_base_joint":
            continue
        if name not in name_to_index:
            raise KeyError(f"Animation joint '{name}' not found in robot articulation. Available: {list(name_to_index.keys())}")
        anim_joint_indices.append(qposadr)
        robot_joint_indices.append(name_to_index[name])

    return anim_joint_indices, robot_joint_indices


def write_frame_to_sim(
    scene: InteractiveScene,
    frame: List[float],
    base_pos_idx: List[int],
    base_quat_idx: List[int],
    anim_joint_indices: List[int],
    robot_joint_indices: List[int],
) -> None:
    device = scene["Robot"].device

    # Root pose from animation
    base_pos = torch.tensor([float(frame[base_pos_idx[0]]), float(frame[base_pos_idx[1]]), float(frame[base_pos_idx[2]])], device=device, dtype=torch.float32)
    base_quat_wxyz = torch.tensor([
        float(frame[base_quat_idx[0]]),
        float(frame[base_quat_idx[1]]),
        float(frame[base_quat_idx[2]]),
        float(frame[base_quat_idx[3]]),
    ], device=device, dtype=torch.float32)

    origin = scene.env_origins.to(device=device)[0] if hasattr(scene, "env_origins") else torch.zeros(3, device=device)
    new_root_state = torch.zeros(1, 13, device=device)
    new_root_state[0, :3] = base_pos + origin
    new_root_state[0, 3:7] = base_quat_wxyz / torch.linalg.norm(base_quat_wxyz)
    new_root_state[0, 7:13] = 0.0
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])

    # Joint positions
    joint_positions = scene["Robot"].data.default_joint_pos.clone().to(device=device)
    joint_velocities = torch.zeros_like(joint_positions)

    for a_idx, r_idx in zip(anim_joint_indices, robot_joint_indices):
        joint_positions[0, r_idx] = float(frame[a_idx])

    scene["Robot"].write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()


class PlayerState:
    def __init__(self, total_frames: int, fps: float):
        self.playing: bool = True
        self.direction: int = 1  # 1 forward, -1 backward
        self.speed: float = 1.0  # playback speed multiplier
        self.current: int = 0
        self.start: int = 0
        self.end: int = max(0, total_frames - 1)
        self.dt_frame: float = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 60.0
        self.accum: float = 0.0


class AnimationUI:
    def __init__(self, state: PlayerState, max_frame: int):
        self.state = state
        self._window = ui.Window("Animation Player", width=360, height=260)
        self._max = max_frame

        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Playback Controls", style={"font_size": 16, "color": 0xFFFFFFFF})
                with ui.HStack(spacing=8):
                    self._play_btn = ui.Button("Pause" if self.state.playing else "Play", clicked_fn=self._toggle_play)
                    self._dir_btn = ui.Button("Forward" if self.state.direction == 1 else "Backward", clicked_fn=self._toggle_dir)
                with ui.HStack(spacing=8):
                    ui.Label("Speed:")
                    self._speed_field = ui.FloatField(value=self.state.speed)
                    self._speed_field.model.add_value_changed_fn(self._on_speed_changed)
                with ui.HStack(spacing=8):
                    ui.Label("Start:")
                    self._start_slider = ui.IntSlider(min=0, max=self._max, value=self.state.start)
                    self._start_slider.model.add_value_changed_fn(self._on_start_changed)
                with ui.HStack(spacing=8):
                    ui.Label("End:")
                    self._end_slider = ui.IntSlider(min=0, max=self._max, value=self.state.end)
                    self._end_slider.model.add_value_changed_fn(self._on_end_changed)
                with ui.HStack(spacing=8):
                    ui.Label("Frame:")
                    self._frame_label = ui.Label(f"{self.state.current}/{self._max}")
                with ui.HStack(spacing=8):
                    ui.Button("<< 10", clicked_fn=lambda: self._skip(-10))
                    ui.Button("< 1", clicked_fn=lambda: self._skip(-1))
                    ui.Button("> 1", clicked_fn=lambda: self._skip(1))
                    ui.Button(">> 10", clicked_fn=lambda: self._skip(10))

    def refresh(self) -> None:
        if self._frame_label:
            self._frame_label.text = f"{self.state.current}/{self._max}"
        if self._play_btn:
            self._play_btn.text = "Pause" if self.state.playing else "Play"
        if self._dir_btn:
            self._dir_btn.text = "Forward" if self.state.direction == 1 else "Backward"

    def _toggle_play(self):
        self.state.playing = not self.state.playing
        self.refresh()

    def _toggle_dir(self):
        self.state.direction *= -1
        self.refresh()

    def _on_speed_changed(self, m):
        try:
            val = float(m.as_float)
        except Exception:
            return
        self.state.speed = max(0.0, val)

    def _on_start_changed(self, m):
        v = int(m.as_int)
        self.state.start = max(0, min(v, self.state.end))
        if self.state.current < self.state.start:
            self.state.current = self.state.start
        self.refresh()

    def _on_end_changed(self, m):
        v = int(m.as_int)
        self.state.end = max(self.state.start, min(v, self._max))
        if self.state.current > self.state.end:
            self.state.current = self.state.end
        self.refresh()

    def _skip(self, n: int):
        if n == 0:
            return
        rng = self.state.end - self.state.start + 1
        if rng <= 0:
            return
        rel = (self.state.current - self.state.start + n) % rng
        self.state.current = self.state.start + rel
        self.refresh()


def run_animation_player(sim: sim_utils.SimulationContext, scene: InteractiveScene, anim_path: str) -> None:
    data = load_animation_json(anim_path)
    frames: List[List[float]] = data["frames"]
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("'frames' must be a non-empty list")

    fps = float(data.get("fps", 0.0))
    state = PlayerState(total_frames=len(frames), fps=fps if fps > 0 else 60.0)
    ui_panel = AnimationUI(state, max_frame=len(frames) - 1)

    name_to_index = build_joint_name_to_index_map(scene["Robot"])
    anim_joint_indices, robot_joint_indices = build_animation_index_mapping(data["metadata"], name_to_index)
    base_pos_idx: List[int] = [int(i) for i in data["metadata"]["base"]["pos_indices"]]
    base_quat_idx: List[int] = [int(i) for i in data["metadata"]["base"]["quat_indices"]]

    # Keyboard controls (optional)
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    keys_down = {"SPACE": False, "B": False, "LEFT": False, "RIGHT": False, "LEFT_BRACKET": False, "RIGHT_BRACKET": False, "HOME": False, "END": False, "ESCAPE": False}

    def on_keyboard_event(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key not in keys_down or keys_down[key]:
                return
            keys_down[key] = True
            if key == "SPACE":
                state.playing = not state.playing
                ui_panel.refresh()
            elif key == "B":
                state.direction *= -1
                ui_panel.refresh()
            elif key == "LEFT":
                ui_panel._skip(-1)
            elif key == "RIGHT":
                ui_panel._skip(1)
            elif key == "LEFT_BRACKET":
                span = max(1, (state.end - state.start + 1) // 10)
                ui_panel._skip(-span)
            elif key == "RIGHT_BRACKET":
                span = max(1, (state.end - state.start + 1) // 10)
                ui_panel._skip(span)
            elif key == "HOME":
                state.current = state.start
                ui_panel.refresh()
            elif key == "END":
                state.current = state.end
                ui_panel.refresh()
            elif key == "ESCAPE":
                keys_down["ESCAPE"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_down:
                keys_down[event.input.name] = False

    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # Initial apply
    write_frame_to_sim(
        scene,
        frames[state.current],
        base_pos_idx,
        base_quat_idx,
        anim_joint_indices,
        robot_joint_indices,
    )
    sim.step()
    scene.update(sim.get_physics_dt())

    try:
        sim_dt = sim.get_physics_dt()
        while simulation_app.is_running():
            if keys_down["ESCAPE"]:
                break

            # Advance time/frame
            if state.playing and state.speed > 0.0:
                state.accum += sim_dt * state.speed
                while state.accum >= state.dt_frame:
                    state.accum -= state.dt_frame
                    nxt = state.current + state.direction
                    if nxt > state.end:
                        nxt = state.start
                    elif nxt < state.start:
                        nxt = state.end
                    state.current = nxt

            # Apply current frame
            write_frame_to_sim(
                scene,
                frames[state.current],
                base_pos_idx,
                base_quat_idx,
                anim_joint_indices,
                robot_joint_indices,
            )
            ui_panel.refresh()

            sim.step()
            scene.update(sim_dt)
    finally:
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Scene
    scene_cfg_class = create_scene_cfg()
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    anim_path = "assets/animation_rc4.json"
    print(f"[INFO]: Loading animation from '{anim_path}'...")
    run_animation_player(sim, scene, anim_path)


if __name__ == "__main__":
    main()
    simulation_app.close()




