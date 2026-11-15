import math
import threading
import time

import numpy as np
import pytest

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner
from a2a_robots_lerobot import (
    SO_ARM_JOINTS,
    AnalyticArmKinematics,
    ArmSpec,
    FakeLeRobot,
    LeRobotArmBackend,
    LeRobotKinematicsPlanner,
    matrix_to_pose,
    pose_error,
    pose_to_matrix,
)


def _offset_pose(pose: Pose, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
    x, y, z = pose.position
    return Pose(pose.frame, (x + dx, y + dy, z + dz), pose.orientation)


def test_pose_matrix_round_trip():
    original = Pose("world", (0.25, -0.18, 0.14), (0.92388, 0.0, 0.382683, 0.0))
    mat = pose_to_matrix(original)
    restored = matrix_to_pose(mat, "world")
    pos_err, ori_err = pose_error(original, restored)
    assert pos_err < 1e-6
    assert ori_err < 1e-5


def test_pose_to_matrix_rejects_zero_quaternion():
    with pytest.raises(ValueError, match="zero"):
        pose_to_matrix(Pose("world", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))


def test_plan_is_speed_limited_and_reaches_target_5dof():
    robot = FakeLeRobot(bimanual=False, dof=5)
    spec = robot.arms["right"]
    obs = robot.get_observation()
    q0 = [math.radians(obs[k]) for k in spec.resolved_observation_keys()]
    start = dict(zip(spec.arm_joints, q0, strict=True))

    max_speed = 0.25
    dt = 0.02
    planner = LeRobotKinematicsPlanner(
        robot.arms, solvers=robot.kinematics, max_speed_rad_s=max_speed
    )
    start_pose = planner.forward_pose("right", q0)
    target = _offset_pose(start_pose, dx=0.02, dz=0.01)

    trajectory = planner.plan("right", start, target, dt)

    assert trajectory.joints == SO_ARM_JOINTS
    assert trajectory.dt == dt
    assert len(trajectory.points) >= 5

    points = [tuple(q0)] + list(trajectory.points)
    for a, b in zip(points, points[1:], strict=False):
        max_step = max(abs(vb - va) for va, vb in zip(a, b, strict=True))
        assert max_step / dt <= max_speed * 1.01

    reached = planner.forward_pose("right", trajectory.points[-1])
    pos_err, _ = pose_error(reached, target)
    assert pos_err <= 0.005


def test_plan_checks_orientation_when_enabled():
    robot = FakeLeRobot(bimanual=False, dof=6)
    spec = robot.arms["right"]
    obs = robot.get_observation()
    q0 = [math.radians(obs[k]) for k in spec.resolved_observation_keys()]
    start = dict(zip(spec.arm_joints, q0, strict=True))

    planner = LeRobotKinematicsPlanner(robot.arms, solvers=robot.kinematics)
    start_pose = planner.forward_pose("right", q0)
    target = _offset_pose(start_pose, dx=0.015, dz=0.01)

    trajectory = planner.plan("right", start, target, 0.02)
    reached = planner.forward_pose("right", trajectory.points[-1])
    pos_err, ori_err = pose_error(reached, target)
    assert pos_err <= 0.005
    assert ori_err <= 0.05


def test_plan_rejects_large_joint_step():
    robot = FakeLeRobot(bimanual=False, dof=5)
    spec = robot.arms["right"]
    obs = robot.get_observation()
    q0 = [math.radians(obs[k]) for k in spec.resolved_observation_keys()]
    start = dict(zip(spec.arm_joints, q0, strict=True))

    planner = LeRobotKinematicsPlanner(
        robot.arms, solvers=robot.kinematics, max_joint_delta_rad=0.005
    )
    start_pose = planner.forward_pose("right", q0)
    target = _offset_pose(start_pose, dx=-0.02, dz=0.02)

    with pytest.raises(PlanError) as exc_info:
        planner.plan("right", start, target, 0.02)
    assert exc_info.value.failure_code is FailureCode.UNREACHABLE
    assert "joint step" in exc_info.value.detail


def test_plan_applies_base_transform_for_bimanual():
    robot = FakeLeRobot(bimanual=True, dof=5)
    planner = LeRobotKinematicsPlanner(robot.arms, solvers=robot.kinematics)

    obs = robot.get_observation()
    q_left = [math.radians(obs[k]) for k in robot.arms["left"].resolved_observation_keys()]
    q_right = [math.radians(obs[k]) for k in robot.arms["right"].resolved_observation_keys()]
    left_pose = planner.forward_pose("left", q_left)
    right_pose = planner.forward_pose("right", q_right)

    assert left_pose.position[1] > 0.1
    assert right_pose.position[1] < -0.1
    assert left_pose.position[0] == pytest.approx(right_pose.position[0], abs=1e-6)


def test_unreachable_target_raises_plan_error():
    robot = FakeLeRobot(bimanual=False, dof=5)
    spec = robot.arms["right"]
    obs = robot.get_observation()
    q0 = [math.radians(obs[k]) for k in spec.resolved_observation_keys()]
    start = dict(zip(spec.arm_joints, q0, strict=True))
    planner = LeRobotKinematicsPlanner(robot.arms, solvers=robot.kinematics)

    with pytest.raises(PlanError) as exc_info:
        planner.plan("right", start, Pose("world", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), 0.02)
    assert exc_info.value.failure_code is FailureCode.UNREACHABLE


def test_realtime_move_paces_control_steps():
    robot = FakeLeRobot(bimanual=False, dof=5)
    backend = LeRobotArmBackend(
        robot, robot.arms, solvers=robot.kinematics, realtime=True, max_speed_rad_s=0.25
    )
    start_pose = backend.observe()["right"].pose
    target = _offset_pose(start_pose, dz=0.01)

    t0 = time.monotonic()
    outcome = backend.move_to("right", target, threading.Event())
    elapsed = time.monotonic() - t0

    assert outcome.failure_code is None
    assert elapsed >= 0.08


class ScriptedPlanner(Planner):
    def __init__(self, trajectory: JointTrajectory) -> None:
        self.trajectory = trajectory
        self.calls = 0

    def plan(self, hand, start, target, dt):
        del hand, start, target, dt
        self.calls += 1
        return self.trajectory


def test_backend_uses_custom_planner_and_validates_joint_names():
    robot = FakeLeRobot(bimanual=False, dof=5)
    bad_planner = ScriptedPlanner(JointTrajectory(("wrong_joint",), ((0.0,),), 0.02))
    backend = LeRobotArmBackend(
        robot,
        robot.arms,
        solvers=robot.kinematics,
        realtime=False,
        planner=bad_planner,
    )
    target = backend.observe()["right"].pose

    with pytest.raises(ValueError, match="expected"):
        backend.move_to("right", target, threading.Event())
    assert bad_planner.calls == 1


def test_arm_spec_validation():
    with pytest.raises(ValueError, match="arm_joints"):
        ArmSpec(arm_joints=())
    with pytest.raises(ValueError, match="unique"):
        ArmSpec(arm_joints=("j1", "j1"))
    with pytest.raises(ValueError, match="joint_unit"):
        ArmSpec(arm_joints=("j1",), joint_unit="turns")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_transform"):
        ArmSpec(arm_joints=("j1",), base_transform=np.eye(3))
    with pytest.raises(ValueError, match="closed_threshold"):
        ArmSpec(arm_joints=("j1",), closed_threshold=98.0, open_threshold=95.0)
    with pytest.raises(ValueError, match="urdf_path"):
        LeRobotKinematicsPlanner({"right": ArmSpec(arm_joints=("j1",))})


def test_analytic_kinematics_validation():
    kin = AnalyticArmKinematics(dof=5)
    with pytest.raises(ValueError, match="expected 5"):
        kin.forward_kinematics([0.0, 0.0])
    with pytest.raises(ValueError, match="4x4"):
        kin.inverse_kinematics([0.0] * 5, np.eye(3))
