"""
Interactive control: while the simulation is running, type any chess square
(e.g. "e4") and press Enter, and the robot will drive straight there. Type a
new square at any time to redirect it, even mid-move.

Usage:
    python interactive_drive.py

Then in the terminal:
    > e4
    > a1
    > h8
    > quit          (stops taking input; close the viewer window to exit)
"""

import threading
import time

import numpy as np
import mujoco
import mujoco.viewer

from drive_robot import square_to_xy


class LiveTargetController:
    """Drives the robot toward a target that can be changed at any time."""

    def __init__(self, start_xy, kp=8.0, kd=5.0, kp_yaw=3.0, kd_yaw=1.0):
        self.target = start_xy
        self.lock = threading.Lock()
        self.kp, self.kd = kp, kd
        self.kp_yaw, self.kd_yaw = kp_yaw, kd_yaw

    def set_target(self, xy):
        with self.lock:
            self.target = xy

    def step(self, data):
        with self.lock:
            tx, ty = self.target

        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, vyaw = data.qvel[0], data.qvel[1], data.qvel[2]

        ex, ey = tx - x, ty - y
        dist = np.hypot(ex, ey)

        data.ctrl[0] = self.kp * ex - self.kd * vx
        data.ctrl[1] = self.kp * ey - self.kd * vy

        if dist > 0.02:
            target_yaw = np.arctan2(ey, ex)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            data.ctrl[2] = self.kp_yaw * yaw_err - self.kd_yaw * vyaw
        else:
            data.ctrl[2] = -self.kd_yaw * vyaw


def input_loop(controller):
    """Runs in a background thread, reading square names from the terminal."""
    print("Type a square (e.g. e4) and press Enter. Type 'quit' to stop taking input.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break

        if text.lower() in ("quit", "exit"):
            print("No longer taking input — close the viewer window to fully exit.")
            break
        if not text:
            continue

        try:
            xy = square_to_xy(text)
        except (ValueError, IndexError):
            print(f"'{text}' isn't a valid square — try something like 'e4' (a1-h8).")
            continue

        controller.set_target(xy)
        print(f"Heading to {text.lower()} ...")


def main():
    model = mujoco.MjModel.from_xml_path("chess_grid.xml")
    data = mujoco.MjData(model)

    # Start with the target equal to wherever the robot currently is
    start_xy = (float(data.qpos[0]), float(data.qpos[1]))
    controller = LiveTargetController(start_xy)

    input_thread = threading.Thread(target=input_loop, args=(controller,), daemon=True)
    input_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            controller.step(data)
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
