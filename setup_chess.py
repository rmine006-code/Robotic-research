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
from Chess import Chess, Pawn, Bishop, Knight, King, Queen, create_Chess_board

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

    def step(self, data, rd):
        with self.lock:
            tx, ty = self.target
        x_id, y_id, yaw_id = rd["pos"]
        x_ctrl, y_ctrl, yaw_ctrl = rd["ctrl"]
        x_vel, y_vel, yaw_vel = rd["vel"]

        x, y, yaw = data.qpos[x_id] + rd["spawn"][0], data.qpos[y_id] + rd["spawn"][1], data.qpos[yaw_id]
        vx, vy, vyaw = data.qvel[x_vel], data.qvel[y_vel], data.qvel[yaw_vel]

        ex, ey = tx - x, ty - y
        dist = np.hypot(ex, ey)

        data.ctrl[x_ctrl] = self.kp * ex - self.kd * vx
        data.ctrl[y_ctrl] = self.kp * ey - self.kd * vy

        if dist > 0.02:
            target_yaw = np.arctan2(ey, ex)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            data.ctrl[yaw_ctrl] = self.kp_yaw * yaw_err - self.kd_yaw * vyaw
        else:
            data.ctrl[yaw_ctrl] = -self.kd_yaw * vyaw


def input_loop(controller, obj):
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
    num_instances = 2
    model = mujoco.MjModel.from_xml_path("chess_grid.xml")
    data = mujoco.MjData(model)

    robot_names = ["robot1", "robot2"]

    # Get qpos addresses (for reading/writing position) per robot
    # Get actuator IDs (for sending control signals) per robot
    robot_data = []
    for name in robot_names:
        x_id = model.joint(f"{name}_x").qposadr[0]
        y_id = model.joint(f"{name}_y").qposadr[0]
        yaw_id = model.joint(f"{name}_yaw").qposadr[0]
        x_vel = model.joint(f"{name}_x").dofadr[0]
        y_vel = model.joint(f"{name}_y").dofadr[0]
        yaw_vel = model.joint(f"{name}_yaw").dofadr[0]
        x_ctrl = model.actuator(f"{name}_motor_x").id
        y_ctrl = model.actuator(f"{name}_motor_y").id
        yaw_ctrl = model.actuator(f"{name}_motor_yaw").id
        body_id = model.body(f"{name}").id
        spawn_xy = (float(model.body_pos[body_id][0]), float(model.body_pos[body_id][1]))
        robot_data.append({
            "pos": (x_id, y_id, yaw_id),
            "vel": (x_vel, y_vel, yaw_vel),
            "ctrl": (x_ctrl, y_ctrl, yaw_ctrl),
            "spawn": (spawn_xy)
            })
    chess_pieces = Chess()
    chess_pieces.add_pieces(Knight(0,7,"White"))
    chess_pieces.add_pieces(Bishop(0,0,"Black"))
    arr, empty = create_Chess_board()
    
    # Start with the target equal to wherever the robot currently is
    start_xy = [ga["spawn"] for ga in robot_data]
    controller = [LiveTargetController(xy) for xy in start_xy]
    for i in range(num_instances):
        controller[i].set_target(square_to_xy(chess_pieces.pieces[i].get_space(arr)))
        empty[chess_pieces.pieces[i].x][chess_pieces.pieces[i].y] = False

    input_thread = threading.Thread(target=input_loop, args=(controller[0],chess_pieces.pieces[0],), daemon=True)
    input_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            for i in range(num_instances):
                controller[i].step(data,robot_data[i])
            
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
