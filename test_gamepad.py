import glfw, time

if not glfw.init():
    raise SystemExit("Failed to init GLFW")

jid = glfw.JOYSTICK_1

if not glfw.joystick_present(jid):
    print("No joystick found")
    glfw.terminate()
    raise SystemExit

print("Name:", glfw.get_joystick_name(jid))

if glfw.joystick_is_gamepad(jid):
    print("Using gamepad mapping:", glfw.get_gamepad_name(jid))
else:
    print("Not a standard gamepad mapping")

print("Move sticks / press buttons (Ctrl+C to exit)")

try:
    while True:
        state = glfw.get_gamepad_state(jid)
        if state:
            # Axes are floats [-1,1], buttons are 0/1
            axes = [round(a, 2) for a in state.axes]
            buttons = state.buttons
            print(f"axes={axes} buttons={buttons}", end="\r", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nExiting…")
finally:
    glfw.terminate()
