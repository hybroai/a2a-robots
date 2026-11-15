"""Integration tests against real ``lerobot`` classes and ``PlacoKinematics``.

Skipped when ``lerobot`` is not installed unless ``A2A_ROBOTS_REQUIRE_LEROBOT=1``
is set.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("A2A_ROBOTS_REQUIRE_LEROBOT") == "1":
    import lerobot
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots import Robot
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
else:
    lerobot = pytest.importorskip("lerobot")
    pytest.importorskip("placo")
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots import Robot
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from a2a_robots import Pose, RobotAdapter, SafetyLimits, Workspace  # noqa: E402
from a2a_robots.skills import parse_request  # noqa: E402
from a2a_robots_lerobot import (  # noqa: E402
    SO_ARM_JOINTS,
    FakeLeRobot,
    LeRobotArmBackend,
    PlacoKinematics,
    bi_so_arms,
    koch_arms,
    matrix_to_pose,
    so101_arms,
)

# Minimal 5-DoF URDF using the exact SO-101 joint and frame names.
_SO101_TEST_URDF = """<?xml version="1.0"?>
<robot name="so101_test">
  <link name="base_link"/>
  <link name="link_1"/>
  <link name="link_2"/>
  <link name="link_3"/>
  <link name="link_4"/>
  <link name="link_5"/>
  <link name="gripper_frame_link"/>

  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
  </joint>
  <joint name="shoulder_lift" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
  </joint>
  <joint name="elbow_flex" type="revolute">
    <parent link="link_2"/>
    <child link="link_3"/>
    <origin xyz="0.18 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
  </joint>
  <joint name="wrist_flex" type="revolute">
    <parent link="link_3"/>
    <child link="link_4"/>
    <origin xyz="0.16 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
  </joint>
  <joint name="wrist_roll" type="revolute">
    <parent link="link_4"/>
    <child link="link_5"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
  </joint>
  <joint name="gripper_fixed" type="fixed">
    <parent link="link_5"/>
    <child link="gripper_frame_link"/>
    <origin xyz="0.06 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


class _RealSubclassRobot(Robot):
    """Concrete ``lerobot.robots.Robot`` subclass using SO-101 config & features."""

    config_class = SO101FollowerConfig
    name = "test_so101_subclass"
    supports_position_hold = True

    def __init__(self, config: SO101FollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self._delegate = FakeLeRobot(bimanual=False, use_degrees=config.use_degrees, dof=5)

    @property
    def observation_features(self) -> dict:
        return self._delegate.observation_features

    @property
    def action_features(self) -> dict:
        return self._delegate.action_features

    @property
    def is_connected(self) -> bool:
        return self._delegate.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._delegate.connect(calibrate=calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self._delegate.is_calibrated

    def calibrate(self) -> None:
        self._delegate.calibrate()

    def configure(self) -> None:
        pass

    def get_observation(self):
        return self._delegate.get_observation()

    def send_action(self, action):
        return self._delegate.send_action(action)

    def disconnect(self) -> None:
        self._delegate.disconnect()


def test_real_lerobot_so101_features_match_presets():
    assert hasattr(lerobot, "__version__")
    so_cfg = SO101FollowerConfig(port="/dev/null", use_degrees=True)
    so_robot = SO101Follower(so_cfg)
    arms = so101_arms()
    spec = arms["right"]
    assert set(spec.resolved_action_keys()) <= set(so_robot.action_features)
    assert spec.gripper_action_key in so_robot.action_features

    bi_cfg = BiSOFollowerConfig(
        left_arm_config=SO101FollowerConfig(port="/dev/null", use_degrees=True),
        right_arm_config=SO101FollowerConfig(port="/dev/null", use_degrees=True),
    )
    bi_robot = BiSOFollower(bi_cfg)
    bi_arms = bi_so_arms()
    for hand_spec in bi_arms.values():
        assert set(hand_spec.resolved_action_keys()) <= set(bi_robot.action_features)
        assert hand_spec.gripper_action_key in bi_robot.action_features

    koch_robot = KochFollower(KochFollowerConfig(port="/dev/null", use_degrees=True))
    koch_spec = koch_arms()["right"]
    assert set(koch_spec.resolved_action_keys()) <= set(koch_robot.action_features)
    assert koch_spec.gripper_action_key in koch_robot.action_features


async def test_placo_kinematics_and_real_robot_subclass_end_to_end(tmp_path: Path):
    urdf_path = tmp_path / "so101_test.urdf"
    urdf_path.write_text(_SO101_TEST_URDF)

    # Verify RobotKinematics itself works with the generated URDF.
    rk = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=list(SO_ARM_JOINTS),
    )
    pk = PlacoKinematics(
        urdf_path, target_frame_name="gripper_frame_link", joint_names=SO_ARM_JOINTS
    )
    q_deg = np.array([0.0, -20.0, 40.0, 10.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(
        pk.forward_kinematics(np.deg2rad(q_deg)),
        rk.forward_kinematics(q_deg),
        atol=1e-6,
    )

    robot = _RealSubclassRobot(
        SO101FollowerConfig(port="/dev/null", id="test_arm", use_degrees=True)
    )
    arms = so101_arms(urdf_path)
    backend = LeRobotArmBackend(robot, arms, realtime=False)
    limits = SafetyLimits(
        workspace=Workspace("world", (-0.5, -0.5, -0.2), (0.6, 0.5, 0.6)),
        max_step_m=0.05,
        max_step_rad=0.5,
    )
    adapter = RobotAdapter(backend, limits, name="SO101 Placo")

    obs_before = backend.observe()["right"].pose
    target = Pose(
        "world",
        (
            obs_before.position[0] + 0.015,
            obs_before.position[1],
            obs_before.position[2] + 0.01,
        ),
        obs_before.orientation,
    )
    result = await adapter.controller.run(
        parse_request(
            {
                "skill": "move_to",
                "args": {
                    "hand": "right",
                    "pose": target.to_dict(),
                },
            },
            backend.hands,
        ),
        "move-1",
    )
    assert result.success is True
    reached = matrix_to_pose(
        rk.forward_kinematics(
            np.array(
                [robot.get_observation()[k] for k in arms["right"].resolved_observation_keys()]
            )
        ),
        "world",
    )
    assert math.dist(reached.position, target.position) <= 0.005

    outcome = backend.open("right", threading.Event())
    assert outcome.failure_code is None
    await adapter.close()
    assert robot.is_connected is False
