"""Integration tests against real ``reachy2-sdk`` classes over gRPC and optional Docker sim.

Skipped when ``reachy2_sdk`` is not installed unless ``A2A_ROBOTS_REQUIRE_REACHY2_SDK=1``
is set (as in the dedicated CI test step). Set ``REACHY2_HOST`` (and optionally
``REACHY2_PORT``) to also run against Pollen's Reachy 2 Docker simulator.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent import futures

import numpy as np
import pytest

if os.environ.get("A2A_ROBOTS_REQUIRE_REACHY2_SDK") == "1":
    import grpc
    from google.protobuf.empty_pb2 import Empty
    from google.protobuf.timestamp_pb2 import Timestamp
    from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
    from reachy2_sdk import ReachySDK
    from reachy2_sdk.grippers.parallel_gripper import ParallelGripper
    from reachy2_sdk.parts.arm import Arm
    from reachy2_sdk_api import (
        arm_pb2,
        arm_pb2_grpc,
        component_pb2,
        error_pb2,
        goto_pb2,
        goto_pb2_grpc,
        hand_pb2,
        hand_pb2_grpc,
        kinematics_pb2,
        orbita2d_pb2,
        orbita3d_pb2,
        part_pb2,
        reachy_pb2,
        reachy_pb2_grpc,
    )
else:
    pytest.importorskip("reachy2_sdk")
    pytest.importorskip("reachy2_sdk_api")
    import grpc
    from google.protobuf.empty_pb2 import Empty
    from google.protobuf.timestamp_pb2 import Timestamp
    from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
    from reachy2_sdk import ReachySDK
    from reachy2_sdk.grippers.parallel_gripper import ParallelGripper
    from reachy2_sdk.parts.arm import Arm
    from reachy2_sdk_api import (
        arm_pb2,
        arm_pb2_grpc,
        component_pb2,
        error_pb2,
        goto_pb2,
        goto_pb2_grpc,
        hand_pb2,
        hand_pb2_grpc,
        kinematics_pb2,
        orbita2d_pb2,
        orbita3d_pb2,
        part_pb2,
        reachy_pb2,
        reachy_pb2_grpc,
    )

from a2a_robots import Pose  # noqa: E402
from a2a_robots_pollen import FakeReachySDK, Reachy2Backend, connect  # noqa: E402
from a2a_robots_pollen.planner import matrix_to_pose  # noqa: E402

_OPEN_RAD = math.radians(130.0)
_CLOSE_RAD = math.radians(-5.0)


def _orbita2d_state(
    j1_deg: float, j2_deg: float, goal1_deg: float, goal2_deg: float, compliant: bool
) -> orbita2d_pb2.Orbita2dState:
    zero_vec = orbita2d_pb2.Vector2d(x=FloatValue(value=0.0), y=FloatValue(value=0.0))
    unit_f2 = orbita2d_pb2.Float2d(motor_1=FloatValue(value=1.0), motor_2=FloatValue(value=1.0))
    gains = component_pb2.PIDGains(
        p=FloatValue(value=1.0), i=FloatValue(value=0.0), d=FloatValue(value=0.0)
    )
    return orbita2d_pb2.Orbita2dState(
        compliant=BoolValue(value=compliant),
        present_position=orbita2d_pb2.Pose2d(
            axis_1=FloatValue(value=math.radians(j1_deg)),
            axis_2=FloatValue(value=math.radians(j2_deg)),
        ),
        goal_position=orbita2d_pb2.Pose2d(
            axis_1=FloatValue(value=math.radians(goal1_deg)),
            axis_2=FloatValue(value=math.radians(goal2_deg)),
        ),
        present_speed=zero_vec,
        present_load=zero_vec,
        temperature=unit_f2,
        speed_limit=unit_f2,
        torque_limit=unit_f2,
        pid=orbita2d_pb2.PID2d(motor_1=gains, motor_2=gains),
    )


def _orbita3d_state(
    present_deg: tuple[float, float, float],
    goal_deg: tuple[float, float, float],
    compliant: bool,
) -> orbita3d_pb2.Orbita3dState:
    zero_vec = orbita3d_pb2.Vector3d(
        x=FloatValue(value=0.0), y=FloatValue(value=0.0), z=FloatValue(value=0.0)
    )
    unit_f3 = orbita3d_pb2.Float3d(
        motor_1=FloatValue(value=1.0), motor_2=FloatValue(value=1.0), motor_3=FloatValue(value=1.0)
    )
    gains = component_pb2.PIDGains(
        p=FloatValue(value=1.0), i=FloatValue(value=0.0), d=FloatValue(value=0.0)
    )
    return orbita3d_pb2.Orbita3dState(
        compliant=BoolValue(value=compliant),
        present_position=kinematics_pb2.Rotation3d(
            rpy=kinematics_pb2.ExtEulerAngles(
                roll=FloatValue(value=math.radians(present_deg[0])),
                pitch=FloatValue(value=math.radians(present_deg[1])),
                yaw=FloatValue(value=math.radians(present_deg[2])),
            )
        ),
        goal_position=kinematics_pb2.Rotation3d(
            rpy=kinematics_pb2.ExtEulerAngles(
                roll=FloatValue(value=math.radians(goal_deg[0])),
                pitch=FloatValue(value=math.radians(goal_deg[1])),
                yaw=FloatValue(value=math.radians(goal_deg[2])),
            )
        ),
        present_speed=zero_vec,
        present_load=zero_vec,
        temperature=unit_f3,
        speed_limit=unit_f3,
        torque_limit=unit_f3,
        pid=orbita3d_pb2.PID3d(motor_1=gains, motor_2=gains, motor_3=gains),
    )


def _arm_state(arm: object, part_id: part_pb2.PartId) -> arm_pb2.ArmState:
    compliant = not arm.is_on()  # type: ignore[attr-defined]
    return arm_pb2.ArmState(
        id=part_id,
        activated=True,
        shoulder_state=_orbita2d_state(
            arm.shoulder.pitch.present_position,  # type: ignore[attr-defined]
            arm.shoulder.roll.present_position,  # type: ignore[attr-defined]
            arm.shoulder.pitch.goal_position,  # type: ignore[attr-defined]
            arm.shoulder.roll.goal_position,  # type: ignore[attr-defined]
            compliant,
        ),
        elbow_state=_orbita2d_state(
            arm.elbow.yaw.present_position,  # type: ignore[attr-defined]
            arm.elbow.pitch.present_position,  # type: ignore[attr-defined]
            arm.elbow.yaw.goal_position,  # type: ignore[attr-defined]
            arm.elbow.pitch.goal_position,  # type: ignore[attr-defined]
            compliant,
        ),
        wrist_state=_orbita3d_state(
            (
                arm.wrist.roll.present_position,  # type: ignore[attr-defined]
                arm.wrist.pitch.present_position,  # type: ignore[attr-defined]
                arm.wrist.yaw.present_position,  # type: ignore[attr-defined]
            ),
            (
                arm.wrist.roll.goal_position,  # type: ignore[attr-defined]
                arm.wrist.pitch.goal_position,  # type: ignore[attr-defined]
                arm.wrist.yaw.goal_position,  # type: ignore[attr-defined]
            ),
            compliant,
        ),
    )


def _hand_state(gripper: object) -> hand_pb2.HandState:
    opening_frac = gripper.opening / 100.0  # type: ignore[attr-defined]
    goal_frac = gripper.goal_position / 100.0  # type: ignore[attr-defined]
    pos_rad = _CLOSE_RAD + opening_frac * (_OPEN_RAD - _CLOSE_RAD)
    goal_rad = _CLOSE_RAD + goal_frac * (_OPEN_RAD - _CLOSE_RAD)
    return hand_pb2.HandState(
        opening=FloatValue(value=opening_frac),
        present_position=hand_pb2.HandPosition(
            parallel_gripper=hand_pb2.ParallelGripperPosition(position=FloatValue(value=pos_rad))
        ),
        goal_position=hand_pb2.HandPosition(
            parallel_gripper=hand_pb2.ParallelGripperPosition(position=FloatValue(value=goal_rad))
        ),
        compliant=BoolValue(value=not gripper.is_on()),  # type: ignore[attr-defined]
    )


class _FakeReachyServicer(
    reachy_pb2_grpc.ReachyServiceServicer,
    arm_pb2_grpc.ArmServiceServicer,
    hand_pb2_grpc.HandServiceServicer,
    goto_pb2_grpc.GoToServiceServicer,
):
    """In-process Reachy 2 gRPC server backed by ``FakeReachySDK``."""

    def __init__(self, fake: FakeReachySDK) -> None:
        self.fake = fake
        self.lock = threading.Lock()
        self.stop_stream = threading.Event()
        self.send_goal_calls = 0
        self.set_hand_calls = 0
        self.cancel_calls: list[str] = []

    def GetReachy(self, request, context):
        del request, context
        return reachy_pb2.Reachy(
            id=reachy_pb2.ReachyId(id=1, name="reachy"),
            l_arm=arm_pb2.Arm(
                part_id=part_pb2.PartId(id=1, name="l_arm"),
                description=arm_pb2.ArmDescription(
                    shoulder=orbita2d_pb2.Orbita2d(
                        id=component_pb2.ComponentId(id=11, name="l_shoulder"),
                        axis_1=orbita2d_pb2.Axis.PITCH,
                        axis_2=orbita2d_pb2.Axis.ROLL,
                    ),
                    elbow=orbita2d_pb2.Orbita2d(
                        id=component_pb2.ComponentId(id=12, name="l_elbow"),
                        axis_1=orbita2d_pb2.Axis.YAW,
                        axis_2=orbita2d_pb2.Axis.PITCH,
                    ),
                    wrist=orbita3d_pb2.Orbita3d(
                        id=component_pb2.ComponentId(id=13, name="l_wrist"),
                    ),
                ),
            ),
            r_arm=arm_pb2.Arm(
                part_id=part_pb2.PartId(id=2, name="r_arm"),
                description=arm_pb2.ArmDescription(
                    shoulder=orbita2d_pb2.Orbita2d(
                        id=component_pb2.ComponentId(id=21, name="r_shoulder"),
                        axis_1=orbita2d_pb2.Axis.PITCH,
                        axis_2=orbita2d_pb2.Axis.ROLL,
                    ),
                    elbow=orbita2d_pb2.Orbita2d(
                        id=component_pb2.ComponentId(id=22, name="r_elbow"),
                        axis_1=orbita2d_pb2.Axis.YAW,
                        axis_2=orbita2d_pb2.Axis.PITCH,
                    ),
                    wrist=orbita3d_pb2.Orbita3d(
                        id=component_pb2.ComponentId(id=23, name="r_wrist"),
                    ),
                ),
            ),
            l_hand=hand_pb2.Hand(
                part_id=part_pb2.PartId(id=3, name="l_hand"),
                type=hand_pb2.HandType.PARALLEL_GRIPPER,
            ),
            r_hand=hand_pb2.Hand(
                part_id=part_pb2.PartId(id=4, name="r_hand"),
                type=hand_pb2.HandType.PARALLEL_GRIPPER,
            ),
            info=reachy_pb2.ReachyInfo(
                serial_number="reachy2-fake-grpc",
                version_soft="1.7.5",
                api_version="1.0.21",
                core_mode=reachy_pb2.ReachyCoreMode.FAKE,
            ),
        )

    def GetReachyState(self, request, context):
        del request, context
        assert self.fake.l_arm is not None and self.fake.r_arm is not None
        with self.lock:
            return reachy_pb2.ReachyState(
                timestamp=Timestamp(seconds=1),
                id=reachy_pb2.ReachyId(id=1, name="reachy"),
                l_arm_state=_arm_state(self.fake.l_arm, part_pb2.PartId(id=1, name="l_arm")),
                r_arm_state=_arm_state(self.fake.r_arm, part_pb2.PartId(id=2, name="r_arm")),
                l_hand_state=_hand_state(self.fake.l_arm.gripper),
                r_hand_state=_hand_state(self.fake.r_arm.gripper),
            )

    def StreamReachyState(self, request, context):
        while context.is_active() and not self.stop_stream.is_set():
            yield self.GetReachyState(request, context)
            time.sleep(0.005)

    def Audit(self, request, context):
        del request, context
        ok = error_pb2.Error(details="Ok")
        o2 = orbita2d_pb2.Orbita2dStatus(errors=[ok])
        o3 = orbita3d_pb2.Orbita3dStatus(errors=[ok])
        arm_status = arm_pb2.ArmStatus(shoulder_status=o2, elbow_status=o2, wrist_status=o3)
        return reachy_pb2.ReachyStatus(l_arm_status=arm_status, r_arm_status=arm_status)

    def _arm_for_part_id(self, part_id: part_pb2.PartId):
        if part_id.name == "l_arm" or part_id.id == 1:
            assert self.fake.l_arm is not None
            return self.fake.l_arm
        assert self.fake.r_arm is not None
        return self.fake.r_arm

    def _gripper_for_part_id(self, part_id: part_pb2.PartId):
        if part_id.name == "l_hand" or part_id.id == 3:
            assert self.fake.l_arm is not None
            return self.fake.l_arm.gripper
        assert self.fake.r_arm is not None
        return self.fake.r_arm.gripper

    def GetJointPosition(self, request, context):
        del context
        with self.lock:
            q = self._arm_for_part_id(request).get_current_positions(degrees=False)
        return arm_pb2.ArmPosition(
            shoulder_position=orbita2d_pb2.Pose2d(
                axis_1=FloatValue(value=q[0]), axis_2=FloatValue(value=q[1])
            ),
            elbow_position=orbita2d_pb2.Pose2d(
                axis_1=FloatValue(value=q[2]), axis_2=FloatValue(value=q[3])
            ),
            wrist_position=kinematics_pb2.Rotation3d(
                rpy=kinematics_pb2.ExtEulerAngles(
                    roll=FloatValue(value=q[4]),
                    pitch=FloatValue(value=q[5]),
                    yaw=FloatValue(value=q[6]),
                )
            ),
        )

    def ComputeArmFK(self, request, context):
        del context
        pos = request.position
        q_rad = [
            pos.shoulder_position.axis_1.value,
            pos.shoulder_position.axis_2.value,
            pos.elbow_position.axis_1.value,
            pos.elbow_position.axis_2.value,
            pos.wrist_position.rpy.roll.value,
            pos.wrist_position.rpy.pitch.value,
            pos.wrist_position.rpy.yaw.value,
        ]
        with self.lock:
            mat = self._arm_for_part_id(request.id).forward_kinematics(q_rad, degrees=False)
        return arm_pb2.ArmFKSolution(
            success=True,
            end_effector=arm_pb2.ArmEndEffector(
                pose=kinematics_pb2.Matrix4x4(data=mat.flatten().tolist())
            ),
        )

    def ComputeArmIK(self, request, context):
        del context
        q0_pos = request.q0
        q0 = [
            q0_pos.shoulder_position.axis_1.value,
            q0_pos.shoulder_position.axis_2.value,
            q0_pos.elbow_position.axis_1.value,
            q0_pos.elbow_position.axis_2.value,
            q0_pos.wrist_position.rpy.roll.value,
            q0_pos.wrist_position.rpy.pitch.value,
            q0_pos.wrist_position.rpy.yaw.value,
        ]
        target = np.asarray(request.target.pose.data, dtype=np.float64).reshape((4, 4))
        with self.lock:
            try:
                q_sol = self._arm_for_part_id(request.id).inverse_kinematics(
                    target, q0=q0, degrees=False
                )
            except ValueError:
                return arm_pb2.ArmIKSolution(success=False)
        return arm_pb2.ArmIKSolution(
            success=True,
            arm_position=arm_pb2.ArmPosition(
                shoulder_position=orbita2d_pb2.Pose2d(
                    axis_1=FloatValue(value=q_sol[0]), axis_2=FloatValue(value=q_sol[1])
                ),
                elbow_position=orbita2d_pb2.Pose2d(
                    axis_1=FloatValue(value=q_sol[2]), axis_2=FloatValue(value=q_sol[3])
                ),
                wrist_position=kinematics_pb2.Rotation3d(
                    rpy=kinematics_pb2.ExtEulerAngles(
                        roll=FloatValue(value=q_sol[4]),
                        pitch=FloatValue(value=q_sol[5]),
                        yaw=FloatValue(value=q_sol[6]),
                    )
                ),
            ),
        )

    def SendComponentsCommands(self, request, context):
        del context
        assert self.fake.l_arm is not None and self.fake.r_arm is not None
        with self.lock:
            self.send_goal_calls += 1
            for cmd in request.shoulder_command.cmd:
                arm = self.fake.l_arm if cmd.id.id == 11 else self.fake.r_arm
                if cmd.HasField("goal_position"):
                    if cmd.goal_position.HasField("axis_1"):
                        arm.shoulder.pitch.goal_position = math.degrees(
                            cmd.goal_position.axis_1.value
                        )
                    if cmd.goal_position.HasField("axis_2"):
                        arm.shoulder.roll.goal_position = math.degrees(
                            cmd.goal_position.axis_2.value
                        )
            for cmd in request.elbow_command.cmd:
                arm = self.fake.l_arm if cmd.id.id == 12 else self.fake.r_arm
                if cmd.HasField("goal_position"):
                    if cmd.goal_position.HasField("axis_1"):
                        arm.elbow.yaw.goal_position = math.degrees(cmd.goal_position.axis_1.value)
                    if cmd.goal_position.HasField("axis_2"):
                        arm.elbow.pitch.goal_position = math.degrees(cmd.goal_position.axis_2.value)
            for cmd in request.wrist_command.cmd:
                arm = self.fake.l_arm if cmd.id.id == 13 else self.fake.r_arm
                if cmd.HasField("goal_position") and cmd.goal_position.HasField("rpy"):
                    rpy = cmd.goal_position.rpy
                    if rpy.HasField("roll"):
                        arm.wrist.roll.goal_position = math.degrees(rpy.roll.value)
                    if rpy.HasField("pitch"):
                        arm.wrist.pitch.goal_position = math.degrees(rpy.pitch.value)
                    if rpy.HasField("yaw"):
                        arm.wrist.yaw.goal_position = math.degrees(rpy.yaw.value)
            self.fake.l_arm.send_goal_positions(check_positions=False)
            self.fake.r_arm.send_goal_positions(check_positions=False)
        return Empty()

    def TurnOn(self, request, context):
        del context
        with self.lock:
            if request.name in ("l_arm", "r_arm") or request.id in (1, 2):
                self._arm_for_part_id(request).turn_on()
            else:
                self._gripper_for_part_id(request).turn_on()
        return Empty()

    def TurnOff(self, request, context):
        del context
        with self.lock:
            if request.name in ("l_arm", "r_arm") or request.id in (1, 2):
                self._arm_for_part_id(request).turn_off()
            else:
                self._gripper_for_part_id(request).turn_off()
        return Empty()

    def SetSpeedLimit(self, request, context):
        del request, context
        return Empty()

    def SetHandPosition(self, request, context):
        del context
        with self.lock:
            self.set_hand_calls += 1
            gripper = self._gripper_for_part_id(request.id)
            pg = request.position.parallel_gripper
            if pg.HasField("opening_percentage"):
                gripper.set_opening(pg.opening_percentage.value * 100.0)
            elif pg.HasField("position"):
                pct = ((pg.position.value - _CLOSE_RAD) / (_OPEN_RAD - _CLOSE_RAD)) * 100.0
                gripper.set_opening(max(0.0, min(100.0, pct)))
        return Empty()

    def CancelPartAllGoTo(self, request, context):
        del context
        with self.lock:
            self.cancel_calls.append(request.name or str(request.id))
            if request.name in ("l_arm", "r_arm") or request.id in (1, 2):
                self._arm_for_part_id(request).cancel_all_goto()
            else:
                self._gripper_for_part_id(request).cancel_all_goto()
        return goto_pb2.GoToAck(ack=True)


class _ReachyGrpcServer:
    def __init__(self, fake: FakeReachySDK | None = None) -> None:
        self.fake = fake or FakeReachySDK()
        self.servicer = _FakeReachyServicer(self.fake)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        reachy_pb2_grpc.add_ReachyServiceServicer_to_server(self.servicer, self.server)
        arm_pb2_grpc.add_ArmServiceServicer_to_server(self.servicer, self.server)
        hand_pb2_grpc.add_HandServiceServicer_to_server(self.servicer, self.server)
        goto_pb2_grpc.add_GoToServiceServicer_to_server(self.servicer, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self._clients: list[ReachySDK] = []

    def __enter__(self) -> _ReachyGrpcServer:
        self.server.start()
        return self

    def connect(self) -> ReachySDK:
        reachy = connect("127.0.0.1", sdk_port=self.port)
        assert isinstance(reachy, ReachySDK)
        self._clients.append(reachy)
        return reachy

    def __exit__(self, *args: object) -> None:
        for client in self._clients:
            if client.is_connected():
                client.disconnect()
        self.servicer.stop_stream.set()
        self.server.stop(grace=0.5)


def test_real_reachy2_sdk_move_open_close_grasp_and_hold() -> None:
    with _ReachyGrpcServer() as srv:
        reachy = srv.connect()
        assert isinstance(reachy.r_arm, Arm)
        assert isinstance(reachy.r_arm.gripper, ParallelGripper)

        backend = Reachy2Backend(reachy, realtime=True, control_dt=0.01, settle_time_s=0.3)
        start = backend.observe()["right"].pose
        target = Pose(
            position=(start.position[0] + 0.03, start.position[1], start.position[2]),
            orientation=start.orientation,
            frame=start.frame,
        )

        outcome = backend.move_to("right", target, threading.Event())
        assert outcome.failure_code is None
        assert srv.servicer.send_goal_calls > 0

        moved = backend.observe()["right"].pose
        assert math.dist(moved.position, target.position) <= 0.005

        assert backend.open("right", threading.Event()).failure_code is None
        assert reachy.r_arm.gripper.opening >= 99.0
        assert srv.servicer.set_hand_calls > 0

        # Empty-air close closes past closed_opening_pct with held=False.
        assert backend.close("right", threading.Event()).failure_code is None
        assert reachy.r_arm.gripper.opening <= 2.0
        assert backend.observe()["right"].held is False

        # Place an object at the current end-effector position and grasp it.
        assert backend.open("right", threading.Event()).failure_code is None
        srv.fake.place_object("cube", moved.position, opening_pct=40.0)
        grasp_outcome = backend.close("right", threading.Event())
        assert grasp_outcome.failure_code is None
        state = backend.observe()["right"]
        assert state.held is True
        assert state.contact["finger"] > 0.0

        backend.hold()
        assert {"l_arm", "r_arm", "l_hand", "r_hand"}.issubset(srv.servicer.cancel_calls)

        backend.shutdown()
        assert reachy.is_connected() is False


def test_real_reachy2_sdk_forward_kinematics_workaround() -> None:
    """Confirm Arm.forward_kinematics(None, degrees=False) reproduces the reachy2-sdk 1.0.15
    degree-vs-radian bug while Reachy2Backend.observe() computes the true FK pose."""
    with _ReachyGrpcServer() as srv:
        reachy = srv.connect()
        assert reachy.r_arm is not None
        assert "gripper" in reachy.r_arm._actuators

        # Give background stream thread a tick to populate OrbitaJoint caches.
        time.sleep(0.03)
        backend = Reachy2Backend(reachy, realtime=False)
        observed = backend.observe()["right"].pose

        # Arm.forward_kinematics(None, degrees=True) converts cached degrees -> radians properly.
        true_pose = matrix_to_pose(reachy.r_arm.forward_kinematics(None, degrees=True), "world")
        assert math.dist(observed.position, true_pose.position) < 1e-6

        # Calling forward_kinematics(None, degrees=False) sends degree values as radians.
        buggy_pose = matrix_to_pose(reachy.r_arm.forward_kinematics(None, degrees=False), "world")
        assert math.dist(buggy_pose.position, true_pose.position) > 0.1


def test_real_reachy2_sdk_turns_on_compliant_arms() -> None:
    fake = FakeReachySDK()
    fake.turn_off()
    with _ReachyGrpcServer(fake) as srv:
        reachy = srv.connect()
        assert reachy.r_arm is not None and reachy.r_arm.is_off()

        backend = Reachy2Backend(reachy, turn_on=True, realtime=False)
        assert reachy.r_arm.is_on()
        assert reachy.r_arm.gripper is not None and reachy.r_arm.gripper.is_on()
        assert len(backend.observe()) == 2


@pytest.mark.skipif(not os.environ.get("REACHY2_HOST"), reason="REACHY2_HOST is not set")
def test_live_reachy2_docker_sim() -> None:
    """Smoke test against a running Pollen Reachy 2 Docker simulator / robot."""
    host = os.environ["REACHY2_HOST"]
    port_env = os.environ.get("REACHY2_PORT")
    reachy = connect(host, sdk_port=int(port_env)) if port_env else connect(host)
    backend = Reachy2Backend(reachy, realtime=True)
    try:
        # The Docker sim boots at [0]*7 (straight-down singularity); bend first.
        if hasattr(reachy.r_arm, "goto_posture"):
            reachy.r_arm.goto_posture("elbow_90", wait=True)
        obs = backend.observe()
        assert "right" in obs
        start = obs["right"].pose
        target = Pose(
            position=(start.position[0] + 0.03, start.position[1], start.position[2] + 0.02),
            orientation=start.orientation,
            frame=start.frame,
        )
        assert backend.move_to("right", target, threading.Event()).failure_code is None
        assert math.dist(backend.observe()["right"].pose.position, target.position) <= 0.005
        assert backend.open("right", threading.Event()).failure_code is None
        assert reachy.r_arm is not None and reachy.r_arm.gripper is not None
        assert reachy.r_arm.gripper.opening >= 95.0
        assert backend.close("right", threading.Event()).failure_code is None
        assert reachy.r_arm.gripper.opening <= 2.0
        assert backend.observe()["right"].held is False
        backend.hold()
    finally:
        backend.shutdown()
    assert reachy.is_connected() is False
