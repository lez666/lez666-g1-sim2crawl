#!/usr/bin/env python3
"""Simple standalone gamepad tester using GLFW.

Prints button presses/releases and axis movements in real time.

Requirements:
  - glfw (pip install glfw)

Usage:
  - python sim2sim_mj/gamepad_test.py [--index 1-16] [--name <substr>]
  - Press the Start/Menu button to exit (button 9 on standard mapping)
"""

from __future__ import annotations

import sys
import time
import argparse

import glfw


BUTTON_NAMES = {
    0: "A (bottom/cross)",
    1: "B (right/circle)",
    2: "X (left/square)",
    3: "Y (top/triangle)",
    4: "LB (left bumper)",
    5: "RB (right bumper)",
    6: "LT (left trigger)",
    7: "RT (right trigger)",
    8: "Select/Back",
    9: "Start/Menu",
    10: "L3 (left stick click)",
    11: "R3 (right stick click)",
    12: "D-pad Up",
    13: "D-pad Down",
    14: "D-pad Left",
    15: "D-pad Right",
}


AXIS_NAMES = {
    0: "LX",
    1: "LY",
    2: "RX",
    3: "RY",
    4: "LT",
    5: "RT",
}


def _to_str_name(name_obj) -> str:
    try:
        # glfw might return bytes on some platforms
        if isinstance(name_obj, (bytes, bytearray)):
            return name_obj.decode(errors="ignore")
        return str(name_obj)
    except Exception:
        return str(name_obj)


def _enumerate_devices() -> list[tuple[int, str, bool]]:
    first = getattr(glfw, "JOYSTICK_1", 0)
    last = getattr(glfw, "JOYSTICK_LAST", 15)
    devices: list[tuple[int, str, bool]] = []
    for jid in range(first, last + 1):
        if glfw.joystick_present(jid):
            name = _to_str_name(glfw.get_joystick_name(jid))
            is_gamepad = bool(glfw.joystick_is_gamepad(jid))
            devices.append((jid, name, is_gamepad))
    return devices


def _select_joystick(preferred_index_1based: int | None, preferred_name_substr: str | None) -> int:
    devices = _enumerate_devices()
    if not devices:
        raise RuntimeError("No joysticks detected via GLFW")

    print("[GAMEPAD] Detected devices:")
    first = getattr(glfw, "JOYSTICK_1", 0)
    for jid, name, is_gp in devices:
        idx = (jid - first) + 1
        mapping = glfw.get_gamepad_name(jid) if is_gp else "(no standard mapping)"
        mapping_str = _to_str_name(mapping) if is_gp else mapping
        print(f"  [{idx:2d}] {name}  | mapping: {mapping_str}")

    # Index selection has priority if provided
    if preferred_index_1based is not None:
        first = getattr(glfw, "JOYSTICK_1", 0)
        jid = first + (preferred_index_1based - 1)
        if not glfw.joystick_present(jid):
            raise RuntimeError(f"Requested joystick index {preferred_index_1based} not present")
        if not glfw.joystick_is_gamepad(jid):
            raise RuntimeError(f"Requested joystick index {preferred_index_1based} does not have a standard gamepad mapping")
        return jid

    # Name substring selection
    if preferred_name_substr:
        substr = preferred_name_substr.lower()
        for jid, name, is_gp in devices:
            if substr in name.lower() and is_gp:
                return jid
        raise RuntimeError(f"No standard-mapped gamepad matched name substring: '{preferred_name_substr}'")

    # Default: first standard-mapped gamepad
    for jid, name, is_gp in devices:
        if is_gp:
            return jid
    raise RuntimeError("Found joysticks but none with a standard gamepad mapping")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLFW gamepad tester")
    parser.add_argument("--index", type=int, default=None, help="Preferred joystick index (1-16)")
    parser.add_argument("--name", type=str, default=None, help="Preferred gamepad name substring (case-insensitive)")
    args = parser.parse_args()

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    joystick_id = _select_joystick(args.index, args.name)

    connected_name = _to_str_name(glfw.get_joystick_name(joystick_id))
    print(f"[GAMEPAD] Connected: {connected_name}")

    if glfw.joystick_is_gamepad(joystick_id):
        mapping_name = _to_str_name(glfw.get_gamepad_name(joystick_id))
        print(f"[GAMEPAD] Using standard mapping: {mapping_name}")
    else:
        # Should not happen due to selection constraints
        raise RuntimeError("Selected joystick does not have a standard mapping")

    # Get initial state (fail loudly if unavailable)
    state = glfw.get_gamepad_state(joystick_id)
    if not state:
        glfw.terminate()
        raise RuntimeError("Failed to read initial gamepad state")

    last_buttons = list(state.buttons)
    last_axes = list(state.axes)

    print()
    print("Press the Start/Menu button to exit (button 9).\n")

    try:
        while True:
            state = glfw.get_gamepad_state(joystick_id)
            if not state:
                raise RuntimeError("Lost gamepad state (disconnected?)")

            # Buttons (print on changes)
            for i in range(min(len(state.buttons), len(last_buttons))):
                current = state.buttons[i]
                previous = last_buttons[i]
                if current != previous:
                    verb = "pressed" if current == 1 else "released"
                    name = BUTTON_NAMES.get(i, f"Button {i}")
                    print(f"[BTN] {name}: {verb}")
            
            # Axes (print when change exceeds threshold)
            axis_threshold = 0.02
            for i in range(min(len(state.axes), len(last_axes))):
                current = state.axes[i]
                previous = last_axes[i]
                if abs(current - previous) >= axis_threshold:
                    name = AXIS_NAMES.get(i, f"Axis {i}")
                    print(f"[AXIS] {name}: {current:+.3f}")

            # Exit on Start/Menu rising edge
            if len(state.buttons) > 9 and state.buttons[9] == 1 and last_buttons[9] == 0:
                print("\n[GAMEPAD] Exit button pressed - quitting.")
                break

            # Update last states
            last_buttons = list(state.buttons)
            last_axes = list(state.axes)

            time.sleep(0.02)  # ~50 Hz
    finally:
        # Always terminate GLFW
        try:
            glfw.terminate()
        finally:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Allow Ctrl-C exit
        try:
            glfw.terminate()
        finally:
            sys.exit(0)
    except Exception as e:
        try:
            glfw.terminate()
        finally:
            raise


