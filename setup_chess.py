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
import torch
import numpy as np
import mujoco
import mujoco.viewer
from Chess import Chess, Pawn, Bishop, Knight, King, Queen, create_Chess_board
from marl_nav import ActorCritic, build_obs, DELTAS

from drive_robot import square_to_xy
FILES = "abcdefgh"
GOAL_SQUARES = ["h8", "a1"]          # one goal per robot (robot1, robot2)
SETTLE_TOL = 0.08                    # how close = "arrived at this square"


def cell_name(col, row):
    return f"{FILES[col]}{row + 1}"

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


def input_loop(n,goals,goal_lock,size):
        """Type '<robot number> <square>' e.g. '1 e4' to change a goal live."""
        print(f"Set a goal: <robot 1-{n}> <square>, e.g. '1 e4'. Type 'quit' to stop input.")
        while True:
            try:
                parts = input("> ").strip().lower().split()
            except EOFError:
                break
            if parts and parts[0] in ("quit", "exit"):
                break
            if len(parts) != 2 or not parts[0].isdigit() or not 1 <= int(parts[0]) <= n:
                print(f"Usage: <robot 1-{n}> <square>, e.g. '1 e4'")
                continue
            r, sq = int(parts[0]) - 1, parts[1]
            if len(sq) != 2 or sq[0] not in FILES or not sq[1].isdigit() or not 1 <= int(sq[1]) <= size:
                print(f"'{sq}' isn't a valid square (a1-h8).")
                continue
            new_goal = [FILES.index(sq[0]), int(sq[1]) - 1]
            with goal_lock:
                if any(j != r and (goals[j] == new_goal).all() for j in range(n)):
                    print("Another robot already has that goal, pick a different square.")
                    continue
                goals[r] = new_goal
                GOAL_SQUARES[r] = sq
            print(f"Robot {r + 1} goal -> {sq}")
 


def main():
    ckpt = torch.load("policy.pt", map_location="cpu")
    size = ckpt["size"]
    net = ActorCritic(size)
    net.load_state_dict(ckpt["model"])
    net.eval()
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
    # World-space centre of every square, using YOUR square_to_xy (no guessing
    # about its scale/offset).
    centers = np.array([[square_to_xy(cell_name(c, r)) for c in range(size)]
                            for r in range(size)])            # [row, col, 2]
    def world_xy(i):
        sx, sy = robot_data[i]["spawn"]
        return np.array([sx + data.qpos[robot_data[i]["pos"][0]],
                             sy + data.qpos[robot_data[i]["pos"][1]]])
    
    def nearest_cell(xy):
        d = np.linalg.norm(centers - xy, axis=2)
        r, c = np.unravel_index(np.argmin(d), d.shape)
        return np.array([c, r])
    goals = np.array([[FILES.index(s[0]), int(s[1]) - 1] for s in GOAL_SQUARES])
    goal_lock = threading.Lock()
    # Start with the target equal to wherever the robot currently is
    start_xy = [ga["spawn"] for ga in robot_data]
    controller = [LiveTargetController(xy) for xy in start_xy]
    for i in range(num_instances):
        controller[i].set_target(square_to_xy(chess_pieces.pieces[i].get_space(arr)))
        empty[chess_pieces.pieces[i].x][chess_pieces.pieces[i].y] = False
    
    goals = np.array([[FILES.index(s[0]), int(s[1]) - 1] for s in GOAL_SQUARES])
    cells = [nearest_cell(world_xy(i)) for i in range(num_instances)]
    targets = [tuple(centers[c[1], c[0]]) for c in cells]
    for i in range(num_instances):                       # drive to the first square center
        controller[i].set_target(targets[i])
    last_cmd = 0.0

    input_thread = threading.Thread(target=input_loop, args=(num_instances,goals,goal_lock,size), daemon=True)
    input_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            # Everyone has reached their current square -> ask the CNN for the next move.
            settled = all(np.linalg.norm(world_xy(i) - targets[i]) < SETTLE_TOL for i in range(num_instances))
            timed_out = (data.time - last_cmd) > 4.0   # don't stall if a robot stops just short
            if settled or timed_out:
                last_cmd = data.time
                cells = [nearest_cell(world_xy(i)) for i in range(num_instances)]
                with goal_lock:
                    current_goals = goals.copy()
                obs = torch.as_tensor(build_obs(np.array(cells), current_goals, size))
                with torch.no_grad():
                        logits, _ = net(obs)
                acts = logits.argmax(dim=1).numpy()       # greedy at deployment
            
                # Same conflict rules as training: never move two robots into
                # one square or swap them (safety net on top of the policy).
                prop = [np.clip(cells[i] + DELTAS[acts[i]], 0, size - 1) for i in range(num_instances)]
                for i in range(num_instances):
                     for j in range(num_instances):
                        if i != j and ((prop[i] == prop[j]).all() or
                            ((prop[i] == cells[j]).all() and (prop[j] == cells[i]).all())):
                            prop[i] = cells[i]
                for i in range(num_instances):
                    targets[i] = tuple(centers[prop[i][1], prop[i][0]])
                    controller[i].set_target(targets[i])
                    #print("cells", [cell_name(*c) for c in cells],
                                 # "actions", acts.tolist(), "-> goals", GOAL_SQUARES)
            
            for i in range(num_instances):
                controller[i].step(data,robot_data[i])
            
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
