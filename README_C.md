# Chess Grid Robot -- C++ Port

A C++ port of the Python `drive_robot.py`, `interactive_drive.py`, and
`chess_gui.py` scripts, using MuJoCo's native C/C++ API instead of the
Python bindings.

## What's different from the Python version

- **No `pip install`** -- you link directly against the MuJoCo C library and
  build with CMake.
- **3D viewer**: `drive_robot` and `interactive_drive` use GLFW + MuJoCo's
  own `mjv_*`/`mjr_*` rendering functions (the same approach MuJoCo's own
  C++ sample apps use) instead of `mujoco.viewer.launch_passive`.
- **2D GUI**: there's no C++ standard-library GUI toolkit equivalent to
  Tkinter, so `chess_gui.cpp` draws the board and robot with plain
  immediate-mode OpenGL in a GLFW window. It doesn't draw on-screen file/rank
  text labels (that needs a font-rendering library) -- instead it prints the
  current and target square to the terminal whenever the current square
  changes.
- **Square input**: same idea as the Python version -- a background thread
  reads lines from the terminal (`std::getline`) and updates a thread-safe
  target while the render loop keeps running.

## Files

- `common.h` -- shared square-to-coordinate conversion and the two PD
  controllers (`WaypointController` for a fixed tour, `LiveTargetController`
  for a target that can change live), used by all three programs.
- `chess_grid.xml` -- the same MuJoCo model as the Python version, unchanged.
- `drive_robot.cpp` -- port of `drive_robot.py`.
- `interactive_drive.cpp` -- port of `interactive_drive.py`.
- `chess_gui.cpp` -- port of `chess_gui.py`.
- `CMakeLists.txt` -- build configuration for all three.

## Dependencies

1. **MuJoCo SDK** -- download a release from
   https://github.com/google-deepmind/mujoco/releases (get the archive for
   your OS, e.g. `mujoco-3.x.x-windows-x86_64.zip` or the Linux/macOS
   equivalent). Unzip it somewhere, e.g. `C:\mujoco` or `~/mujoco`.
2. **GLFW3**:
   - Ubuntu/Debian: `sudo apt install libglfw3-dev`
   - macOS: `brew install glfw`
   - Windows: easiest via [vcpkg](https://github.com/microsoft/vcpkg):
     `vcpkg install glfw3`
3. **CMake** (3.16+) and a C++17 compiler (MSVC, GCC, or Clang).

## Build

```bash
mkdir build && cd build

# Point CMake at your MuJoCo install (adjust the path):
cmake .. -DCMAKE_PREFIX_PATH=/path/to/mujoco-3.x.x

# On Windows with vcpkg, also add:
#   -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build . --config Release
```

This produces three executables: `drive_robot`, `interactive_drive`, and
`chess_gui` (in `build/` or `build/Release/` on Windows), each with
`chess_grid.xml` copied alongside it.

## Run

```bash
./drive_robot                          # 3D viewer, default tour
./drive_robot --headless               # no viewer, text output
./drive_robot --squares e4 d4 d5 e5    # custom route

./interactive_drive                    # 3D viewer + terminal square input

./chess_gui                            # 2D top-down window + terminal square input
```

## Known trade-offs vs. the Python version

- The input-reading thread blocks on `std::getline`, so after you close a
  viewer/GUI window, the process may need one more Enter press in the
  terminal to fully exit -- same limitation the Python threading version has.
- `chess_gui` skips on-canvas text labels (file/rank letters, current/target
  square) in favor of printing them to the terminal, to avoid pulling in a
  font-rendering dependency for a minimal example.
- `drive_robot` and `interactive_drive`'s mouse camera controls come
  straight from MuJoCo's standard sample app pattern (drag to rotate,
  right-drag to pan, scroll to zoom).

## Ideas for extending this

Same ideas as the Python README apply here -- swapping in real
differential-drive kinematics, adding obstacles, path planning, etc. -- just
implemented against `mjData`/`mjModel` directly instead of the Python
bindings.
