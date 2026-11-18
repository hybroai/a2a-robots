# a2a-robots-lerobot

[Hugging Face LeRobot](https://github.com/huggingface/lerobot) backend for [a2a-robots](../../README.md).

Install from PyPI: `pip install a2a-robots-lerobot`.

## LeRobot backend

`LeRobotArmBackend` adapts single-arm and bimanual manipulators managed by `lerobot.robots.Robot` (for example `SO100Follower`, `SO101Follower`, `BiSOFollower`, and `KochFollower`) to `a2a_robots.RobotBackend` and `a2a_robots.RobotAdapter`.

- **URDF kinematics in radians:** `LeRobotKinematicsPlanner` wraps `lerobot.model.kinematics.RobotKinematics` (Placo/Pinocchio) via `PlacoKinematics`, converting between LeRobot's degree-based joint space and `a2a_robots`'s metre/radian pose contract. Pass `base_transform` in `ArmSpec` (or use `bi_so_arms`) to position bimanual arms in a shared `"world"` frame.
- **5-DoF arms vs. 6-DoF poses:** 5-DoF arms such as SO-100, SO-101, and Koch cannot independently match arbitrary 6-DoF end-effector orientations at every position. The `so101_arms`, `bi_so_arms`, and `koch_arms` presets set `orientation_weight=0.01` and `check_orientation=False` so `move_to` reaches position accurately while aligning orientation best-effort; set `check_orientation=True` for 6-DoF/7-DoF arms.
- **Safety guards:**
  - `connect=True` always connects with `calibrate=False` so missing calibration never blocks a headless server on interactive terminal prompts; uncalibrated robots fail fast at startup.
  - Robots with `use_degrees=False` (normalized `[-100, 100]` joint space) are rejected by default (`require_use_degrees=True`) unless explicitly configured with `joint_unit="rad"`.
  - Every `send_action()` sends the complete `robot.action_features` dictionary so non-target arms, grippers, and extra actuators stay held.
  - Set `disable_torque_on_disconnect=False` on your LeRobot config if disconnecting on shutdown should not drop a raised arm under gravity.

## Usage

`a2a-robots-lerobot` is an independent `uv` project with its own `uv.lock` and optional extras for `lerobot` (`kinematics`, `serial`). Offline tests and `FakeLeRobot` work out of the box without installing `lerobot` or PyTorch.

Live hardware and `PlacoKinematics` require Python `>=3.12`:

```bash
# Offline with FakeLeRobot (no lerobot install needed):
LEROBOT_FAKE=1 uv run --directory packages/a2a-robots-lerobot python examples/serve_so101.py

# Live SO-101 hardware (installs the `serial` extra: lerobot[kinematics,serial-motors]):
uv run --directory packages/a2a-robots-lerobot --python 3.12 --extra serial \
  python examples/serve_so101.py --serial-port /dev/ttyACM0 --urdf /path/to/so101_new_calib.urdf
```

Or in Python:

```python
from a2a_robots import RobotAdapter, SafetyLimits, Workspace, serve_robot
from a2a_robots_lerobot import LeRobotArmBackend, so101_arms
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot = SO101Follower(
    SO101FollowerConfig(
        port="/dev/ttyACM0", id="my_arm", use_degrees=True, disable_torque_on_disconnect=False
    )
)
backend = LeRobotArmBackend(robot, so101_arms("/path/to/so101_new_calib.urdf"), connect=True)
limits = SafetyLimits(workspace=Workspace("world", (-0.5, -0.5, -0.1), (0.6, 0.5, 0.6)))
serve_robot(RobotAdapter(backend, limits, name="LeRobot SO-101"), port=9100)
```

## Tests

```bash
# Offline unit, contract, and adapter tests against FakeLeRobot
uv run --directory packages/a2a-robots-lerobot --extra dev pytest

# Integration tests against real lerobot and PlacoKinematics (Python >=3.12)
A2A_ROBOTS_REQUIRE_LEROBOT=1 uv run --directory packages/a2a-robots-lerobot --python 3.12 \
  --extra serial --extra dev pytest tests/test_lerobot_integration.py
```
