import threading
import time

import numpy as np
import pytest

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner
from a2a_robots_pollen import (
    FakeReachySDK,
    Reachy2Backend,
    Reachy2Planner,
    hand_joints,
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


def test_plan_is_speed_limited_and_reaches_target():
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    joints = hand_joints("right")
    q0 = reachy.r_arm.get_current_positions(degrees=False)
    start = dict(zip(joints, q0, strict=True))
    start_pose = matrix_to_pose(reachy.r_arm.forward_kinematics(q0, degrees=False), "world")
    target = _offset_pose(start_pose, dx=0.02, dz=0.01)

    max_speed = 0.25
    dt = 0.02
    planner = Reachy2Planner({"right": reachy.r_arm}, max_speed_rad_s=max_speed)
    trajectory = planner.plan("right", start, target, dt)

    assert trajectory.joints == joints
    assert trajectory.dt == dt
    assert len(trajectory.points) >= 5

    points = [tuple(q0)] + list(trajectory.points)
    for a, b in zip(points, points[1:], strict=False):
        max_step = max(abs(vb - va) for va, vb in zip(a, b, strict=True))
        assert max_step / dt <= max_speed * 1.01

    final_fk = reachy.r_arm.forward_kinematics(trajectory.points[-1], degrees=False)
    pos_err, ori_err = pose_error(matrix_to_pose(final_fk, "world"), target)
    assert pos_err <= 0.005
    assert ori_err <= 0.05


def test_unreachable_target_raises_plan_error():
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    joints = hand_joints("right")
    start = dict(zip(joints, reachy.r_arm.get_current_positions(degrees=False), strict=True))
    planner = Reachy2Planner({"right": reachy.r_arm})

    with pytest.raises(PlanError) as exc_info:
        planner.plan("right", start, Pose("world", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), 0.02)

    assert exc_info.value.failure_code is FailureCode.UNREACHABLE


def test_realtime_move_paces_control_steps():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=True, max_speed_rad_s=0.25)
    start_pose = backend.observe()["right"].pose
    target = _offset_pose(start_pose, dz=0.01)

    t0 = time.monotonic()
    outcome = backend.move_to("right", target, threading.Event())
    elapsed = time.monotonic() - t0

    assert outcome.failure_code is None
    # Minimum-jerk 7-DoF trajectory for 1 cm takes > 4 steps (>= 0.08 s at 20 ms).
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
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    bad_planner = ScriptedPlanner(
        JointTrajectory(("wrong_joint",), ((0.0,),), 0.02),
    )
    backend = Reachy2Backend(reachy, realtime=False, planner=bad_planner)
    target = backend.observe()["right"].pose

    outcome = backend.move_to("right", target, threading.Event())
    assert bad_planner.calls == 1
    assert outcome.failure_code is FailureCode.UNREACHABLE
    assert "expected" in outcome.detail


def test_backend_executes_valid_custom_planner_trajectory():
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    joints = hand_joints("right")
    q0 = np.array(reachy.r_arm.get_current_positions(degrees=False))
    q_goal = q0.copy()
    q_goal[0] += 0.05
    target = matrix_to_pose(reachy.r_arm.forward_kinematics(q_goal, degrees=False), "world")

    custom = ScriptedPlanner(
        JointTrajectory(
            joints,
            (
                tuple((q0 + 0.5 * (q_goal - q0)).tolist()),
                tuple(q_goal.tolist()),
            ),
            0.02,
        )
    )
    backend = Reachy2Backend(reachy, realtime=False, planner=custom)
    outcome = backend.move_to("right", target, threading.Event())

    assert outcome.failure_code is None
    assert custom.calls == 1
    assert reachy.r_arm.get_current_positions(degrees=False)[0] == pytest.approx(
        float(q_goal[0]), abs=1e-6
    )


def test_fk_failure_during_plan_converts_to_unreachable():
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    joints = hand_joints("right")
    q0 = reachy.r_arm.get_current_positions(degrees=False)
    start = dict(zip(joints, q0, strict=True))
    start_pose = matrix_to_pose(reachy.r_arm.forward_kinematics(q0, degrees=False), "world")

    def broken_fk(*args, **kwargs):
        raise ValueError("No solution found for the given joints")

    reachy.r_arm.forward_kinematics = broken_fk  # type: ignore[method-assign]
    planner = Reachy2Planner({"right": reachy.r_arm})

    with pytest.raises(PlanError) as exc_info:
        planner.plan("right", start, start_pose, 0.02)

    assert exc_info.value.failure_code is FailureCode.UNREACHABLE
