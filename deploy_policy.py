"""
Run the trained CNN policy in MuJoCo. The policy chooses each robot's next
board square; your LiveTargetController drives the disc there.

Run:  python deploy_policy.py          (needs policy.pt from marl_nav.py)

Assumes (from setup_chess.py): robot_data entries with "pos"/"vel"/"ctrl",
plus "spawn", and LiveTargetController.step(data, rd) computing world position
as spawn + qpos (the fix we discussed).
"""
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer

from marl_nav import ActorCritic, build_obs, DELTAS
from drive_robot import square_to_xy
from setup_chess import LiveTargetController

FILES = "abcdefgh"
GOAL_SQUARES = ["h8", "a1"]          # one goal per robot (robot1, robot2)
SETTLE_TOL = 0.08                    # how close = "arrived at this square"


def cell_name(col, row):
    return f"{FILES[col]}{row + 1}"


class SpawnAwareController(LiveTargetController):
    """Same PD control, but position = body spawn + joint offset (world frame),
    so targets from square_to_xy line up with where the robot really is."""

    def step(self, data, rd):
        with self.lock:
            tx, ty = self.target
        x_id, y_id, yaw_id = rd["pos"]
        x_ctrl, y_ctrl, yaw_ctrl = rd["ctrl"]
        x_vel, y_vel, yaw_vel = rd["vel"]
        sx, sy = rd["spawn"]

        x, y = sx + data.qpos[x_id], sy + data.qpos[y_id]
        yaw = data.qpos[yaw_id]
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


def main():
    ckpt = torch.load("policy.pt", map_location="cpu")
    size = ckpt["size"]
    net = ActorCritic(size)
    net.load_state_dict(ckpt["model"])
    net.eval()

    model = mujoco.MjModel.from_xml_path("chess_grid.xml")
    data = mujoco.MjData(model)
    names = ["robot1", "robot2"]
    n = len(names) 

    robot_data = []
    for name in names:
        j = lambda s: model.joint(f"{name}_{s}")
        a = lambda s: model.actuator(f"{name}_motor_{s}").id
        bp = model.body_pos[model.body(name).id]
        robot_data.append({
            "pos": (j("x").qposadr[0], j("y").qposadr[0], j("yaw").qposadr[0]),
            "vel": (j("x").dofadr[0], j("y").dofadr[0], j("yaw").dofadr[0]),
            "ctrl": (a("x"), a("y"), a("yaw")),
            "spawn": (float(bp[0]), float(bp[1])),
        })

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
    cells = [nearest_cell(world_xy(i)) for i in range(n)]
    ctrl = [SpawnAwareController(tuple(world_xy(i))) for i in range(n)]
    targets = [tuple(centers[c[1], c[0]]) for c in cells]
    for i in range(n):                       # drive to the first square center
        ctrl[i].set_target(targets[i])
    last_cmd = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # Everyone has reached their current square -> ask the CNN for the next move.
            settled = all(np.linalg.norm(world_xy(i) - targets[i]) < SETTLE_TOL for i in range(n))
            timed_out = (data.time - last_cmd) > 4.0   # don't stall if a robot stops just short
            if settled or timed_out:
                last_cmd = data.time
                cells = [nearest_cell(world_xy(i)) for i in range(n)]
                obs = torch.as_tensor(build_obs(np.array(cells), goals, size))
                with torch.no_grad():
                    logits, _ = net(obs)
                acts = logits.argmax(dim=1).numpy()       # greedy at deployment

                # Same conflict rules as training: never move two robots into
                # one square or swap them (safety net on top of the policy).
                prop = [np.clip(cells[i] + DELTAS[acts[i]], 0, size - 1) for i in range(n)]
                for i in range(n):
                    for j in range(n):
                        if i != j and ((prop[i] == prop[j]).all() or
                                       ((prop[i] == cells[j]).all() and (prop[j] == cells[i]).all())):
                            prop[i] = cells[i]
                for i in range(n):
                    targets[i] = tuple(centers[prop[i][1], prop[i][0]])
                    ctrl[i].set_target(targets[i])
                print("cells", [cell_name(*c) for c in cells],
                      "actions", acts.tolist(), "-> goals", GOAL_SQUARES)

            for i in range(n):
                ctrl[i].step(data, robot_data[i])
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()