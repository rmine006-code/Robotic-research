"""
2D graphical interface showing the robot's live position mapped onto an
8x8 chess grid, synced to the MuJoCo physics simulation. Type a square and
click "Go" (or press Enter) to drive the robot there.

This does NOT use MuJoCo's 3D viewer at all -- this window *is* the
visualization, drawn with Tkinter. That also sidesteps any 3D-viewer/OpenGL
window issues, since only the physics engine is used here, not the renderer.

Usage:
    python chess_gui.py
"""

import tkinter as tk

import numpy as np
import mujoco

from drive_robot import square_to_xy, xy_to_square
from interactive_drive import LiveTargetController

BOARD_MIN = -4.0
BOARD_MAX = 4.0
BOARD_SPAN = BOARD_MAX - BOARD_MIN  # 8.0 world units across the board
CANVAS_SIZE = 480
SQUARE_PX = CANVAS_SIZE / 8

LIGHT_SQUARE = "#EDE7DA"
DARK_SQUARE = "#2E241A"
ROBOT_COLOR = "#D9342B"
HEADING_COLOR = "#FFFFFF"
HIGHLIGHT_COLOR = "#4CAF50"
LABEL_COLOR = "#8A8A8A"


def world_to_canvas(x, y):
    """Convert world (x, y) to canvas pixel coordinates (rank 8 at the top)."""
    cx = (x - BOARD_MIN) / BOARD_SPAN * CANVAS_SIZE
    cy = CANVAS_SIZE - (y - BOARD_MIN) / BOARD_SPAN * CANVAS_SIZE
    return cx, cy


class ChessGridApp:
    def __init__(self, root, model, data, controller):
        self.root = root
        self.model = model
        self.data = data
        self.controller = controller
        self.current_target_square = xy_to_square(*controller.target)

        root.title("Robot Position on Chess Grid")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        controls = tk.Frame(root)
        controls.pack(pady=(0, 10))

        tk.Label(controls, text="Go to square:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry = tk.Entry(controls, width=6)
        self.entry.pack(side=tk.LEFT)
        self.entry.bind("<Return>", lambda e: self.on_go())
        self.entry.focus()

        tk.Button(controls, text="Go", command=self.on_go).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, font=("Helvetica", 12)).pack(pady=(0, 10))

        self.draw_board()
        self.update_loop()

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                x0, y0 = col * SQUARE_PX, (7 - row) * SQUARE_PX
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                self.canvas.create_rectangle(x0, y0, x0 + SQUARE_PX, y0 + SQUARE_PX,
                                              fill=color, outline="", tags="board")

        for col in range(8):
            self.canvas.create_text(col * SQUARE_PX + SQUARE_PX / 2, CANVAS_SIZE - 8,
                                     text=chr(ord('a') + col), fill=LABEL_COLOR,
                                     font=("Helvetica", 8), tags="board")
        for row in range(8):
            self.canvas.create_text(8, (7 - row) * SQUARE_PX + 10,
                                     text=str(row + 1), fill=LABEL_COLOR,
                                     font=("Helvetica", 8), tags="board")

    def on_go(self):
        text = self.entry.get().strip()
        try:
            xy = square_to_xy(text)
        except (ValueError, IndexError):
            self.status_var.set(f"'{text}' isn't a valid square (try a1-h8)")
            return
        self.controller.set_target(xy)
        self.current_target_square = text.lower()
        self.entry.delete(0, tk.END)

    def update_loop(self):
        # Advance physics to roughly cover the time between GUI frames (~60Hz)
        frame_dt = 0.016
        steps = max(1, int(frame_dt / self.model.opt.timestep))
        for _ in range(steps):
            self.controller.step(self.data)
            mujoco.mj_step(self.model, self.data)

        self.redraw_robot()
        self.root.after(16, self.update_loop)

    def redraw_robot(self):
        self.canvas.delete("robot")

        x, y, yaw = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        cx, cy = world_to_canvas(x, y)
        radius = SQUARE_PX * 0.32

        # Highlight whichever square the robot is currently centered over
        col = int(np.clip((x - BOARD_MIN) // 1.0, 0, 7))
        row = int(np.clip((y - BOARD_MIN) // 1.0, 0, 7))
        hx0, hy0 = col * SQUARE_PX, (7 - row) * SQUARE_PX
        self.canvas.create_rectangle(hx0, hy0, hx0 + SQUARE_PX, hy0 + SQUARE_PX,
                                      outline=HIGHLIGHT_COLOR, width=3, tags="robot")

        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                 fill=ROBOT_COLOR, outline="", tags="robot")

        # Heading tick -- canvas y is flipped relative to world y
        hx = cx + radius * 0.9 * np.cos(yaw)
        hy = cy - radius * 0.9 * np.sin(yaw)
        self.canvas.create_line(cx, cy, hx, hy, fill=HEADING_COLOR, width=3, tags="robot")

        current_square = xy_to_square(x, y)
        self.status_var.set(f"Current square: {current_square}     Target: {self.current_target_square}")


def main():
    model = mujoco.MjModel.from_xml_path("chess_grid.xml")
    data = mujoco.MjData(model)

    start_xy = (float(data.qpos[0]), float(data.qpos[1]))
    controller = LiveTargetController(start_xy)

    root = tk.Tk()
    ChessGridApp(root, model, data, controller)
    root.mainloop()


if __name__ == "__main__":
    main()
