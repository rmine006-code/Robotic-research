"""
Drive the circular robot around the chess grid using standard algebraic
chess notation ("a1" through "h8").

Usage:
    python drive_robot.py                 # interactive viewer, default tour
    python drive_robot.py --headless      # no viewer, prints progress
    python drive_robot.py --squares e4 d4 d5 e5   # custom waypoint list
"""

import argparse
import time

import mujoco
import numpy as np

BOARD_ORIGIN = -4.0   # world x/y coordinate of the board's edge
SQUARE_SIZE = 1.0      # one chess square = 1 world unit
ARRIVAL_THRESHOLD = 0.05  # how close (m) counts as "arrived" at a square


def square_to_xy(square: str) -> tuple[float, float]:
    """Convert algebraic notation like 'e4' to world (x, y) coordinates."""
    square = square.strip().lower()
    col = ord(square[0]) - ord("a")   # 0-7 -> a-h
    row = int(square[1:]) - 1          # 0-7 -> 1-8
    if not (0 <= col <= 7 and 0 <= row <= 7):
        raise ValueError(f"'{square}' is not a valid square on an 8x8 board")
    x = BOARD_ORIGIN + col * SQUARE_SIZE + SQUARE_SIZE / 2
    y = BOARD_ORIGIN + row * SQUARE_SIZE + SQUARE_SIZE / 2
    return x, y


def xy_to_square(x: float, y: float) -> str:
    """Convert world (x, y) back to the nearest algebraic square, for display."""
    col = int(np.clip((x - BOARD_ORIGIN) // SQUARE_SIZE, 0, 7))
    row = int(np.clip((y - BOARD_ORIGIN) // SQUARE_SIZE, 0, 7))
    return f"{chr(ord('a') + col)}{row + 1}"


class WaypointController:
    """Simple PD controller that drives the disc toward a sequence of (x, y) targets."""

    def __init__(self, waypoints_xy, kp=8.0, kd=5.0, kp_yaw=3.0, kd_yaw=1.0):
        self.waypoints = waypoints_xy
        self.index = 0
        self.kp, self.kd = kp, kd
        self.kp_yaw, self.kd_yaw = kp_yaw, kd_yaw

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    @property
    def current_target(self):
        return self.waypoints[self.index] if not self.done else None

    def step(self, data):
        if self.done:
            data.ctrl[:] = 0
            return

        tx, ty = self.waypoints[self.index]
        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, vyaw = data.qvel[0], data.qvel[1], data.qvel[2]

        ex, ey = tx - x, ty - y
        dist = np.hypot(ex, ey)

        # Position control (PD)
        data.ctrl[0] = self.kp * ex - self.kd * vx
        data.ctrl[1] = self.kp * ey - self.kd * vy

        # Face the direction of travel (only if we're not basically there)
        if dist > 0.02:
            target_yaw = np.arctan2(ey, ex)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            data.ctrl[2] = self.kp_yaw * yaw_err - self.kd_yaw * vyaw
        else:
            data.ctrl[2] = -self.kd_yaw * vyaw

        if dist < ARRIVAL_THRESHOLD:
            self.index += 1


def default_tour():
    """A simple lap around the board's outer ring, ending in the center."""
    return ["a1", "a8", "h8", "h1", "a1", "d4", "e4", "e5", "d5"]


def run_headless(model, data, controller, max_seconds=60.0):
    steps = int(max_seconds / model.opt.timestep)
    last_reported = -1
    for i in range(steps):
        controller.step(data)
        mujoco.mj_step(model, data)

        if i % 200 == 0 or controller.index != last_reported:
            last_reported = controller.index
            square = xy_to_square(data.qpos[0], data.qpos[1])
            target = controller.waypoints[controller.index] if not controller.done else "-"
            print(f"t={data.time:5.2f}s  near square={square:>3}  heading to target #{controller.index}")

        if controller.done:
            print(f"Tour complete at t={data.time:.2f}s")
            break


def run_viewer(model, data, controller):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            controller.step(data)
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--squares", nargs="+", default=None,
                         help="waypoint squares, e.g. --squares e4 d4 d5 e5")
    args = parser.parse_args()

    squares = args.squares if args.squares else default_tour()
    waypoints_xy = [square_to_xy(sq) for sq in squares]
    print("Route:", " -> ".join(squares))

    model = mujoco.MjModel.from_xml_path("chess_grid.xml")
    data = mujoco.MjData(model)
    controller = WaypointController(waypoints_xy)

    if args.headless:
        run_headless(model, data, controller)
    else:
        run_viewer(model, data, controller)


if __name__ == "__main__":
    main()