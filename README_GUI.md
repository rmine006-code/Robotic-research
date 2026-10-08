# Circular Robot on a Chess Grid

A round disc robot that slides freely across an 8x8 checkered board and can
be commanded to drive between squares using standard chess notation
(`a1`–`h8`).

## Setup

```bash
pip install mujoco numpy
```

## Files

- `chess_grid.xml` — the world: an 8x8 checkerboard floor (one texture
  square = one world unit = one chess square) plus a low border frame, and
  a circular robot with three joints (slide-x, slide-y, hinge-yaw) so it can
  move and rotate freely in the plane.
- `drive_robot.py` — converts chess squares like `"e4"` into world
  coordinates and uses a PD controller to drive the robot between them,
  rotating to face its direction of travel as it goes.

## Run it

Default tour (loops the board's edge, then finishes in the center):

```bash
python drive_robot.py
```

Custom route:

```bash
python drive_robot.py --squares a1 h1 h8 a8 e4
```

Headless (prints which square the robot is near as it drives):

```bash
python drive_robot.py --headless --squares a1 d4 h8
```

## How the coordinates work

The board is centered at the world origin, spanning `x, y ∈ [-4, 4]`, with
each square exactly 1 unit wide. `square_to_xy()` in `drive_robot.py` maps
`a1` → column 0, row 0 → the square centered at `(-3.5, -3.5)`, all the way
up to `h8` → `(3.5, 3.5)`. The checkerboard texture uses `texrepeat="8 8"`
on the same 8x8 unit plane, so the visual squares line up exactly with the
coordinate grid — square `a1`'s corner sits exactly at the texture square's
corner.

## How the robot moves

The robot isn't a wheeled model here — it's a free-sliding disc with:
- `robot_x`, `robot_y`: slide joints (translation in the plane)
- `robot_yaw`: hinge joint (rotation, shown by the white heading marker)

A simple PD (proportional-derivative) controller in `WaypointController`
pushes the disc toward each waypoint in turn, switching to the next one once
it's within 5cm, and gently steers the yaw to face the direction of travel.

## 2D graphical interface

For a top-down view that makes it obvious exactly which square the robot is
on, use `chess_gui.py`:

```bash
python chess_gui.py
```

This opens a Tkinter window (no separate terminal needed) showing:
- An 8x8 board drawn to scale, labeled `a`-`h` and `1`-`8`.
- A red circle for the robot at its exact live position, with a white tick
  showing its heading.
- A green outline highlighting whichever square the robot is currently
  centered over.
- A text box + "Go" button (or just press Enter) to send the robot to any
  square.

This window *is* the visualization — it runs the physics simulation
internally but never opens MuJoCo's 3D viewer, so it also sidesteps any
3D-viewer/OpenGL windowing issues you might hit with `drive_robot.py` or
`interactive_drive.py`.

## Interactive mode

Want to just type squares and watch the robot go there, live? Use
`interactive_drive.py` instead:

```bash
python interactive_drive.py
```

This opens the viewer and, in the same terminal, prompts you for input:

```
> e4
Heading to e4 ...
> a1
Heading to a1 ...
```

Type any square from `a1` to `h8` and press Enter — the robot immediately
redirects toward it, even if it's still moving toward a previous target.
Type `quit` to stop taking new input (close the viewer window to fully
exit the program).

## Ideas for extending this

1. **Real differential-drive kinematics** — replace the free-slide model with
   two wheel geoms and actuate wheel velocities instead of applying x/y
   forces directly, for a more realistic mobile robot.
2. **Highlight the target square** — add a semi-transparent geom that moves
   to show the current waypoint.
3. **Obstacles / other pieces** — drop cylinder or box "chess pieces" onto
   other squares and have the robot path around them.
4. **Path planning** — instead of a hardcoded list of squares, implement
   A* or BFS on the 8x8 grid to route around obstacles automatically.
5. **Knight's-move only** — constrain `--squares` input (or add a mode) so
   the robot can only move in legal knight jumps, and animate a hop instead
   of a slide.

## Notes

Written without a live MuJoCo install available in this environment (no
network access here), so give it a run locally and let me know if anything
needs adjusting — happy to tweak the PD gains, board size, or robot shape.
