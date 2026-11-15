"""Serve an SO-101 / LeRobot arm on http://127.0.0.1:9100/.

Uses ``FakeLeRobot`` when ``--fake`` is passed or ``LEROBOT_FAKE=1`` is set,
and connects to an ``SO101Follower`` arm on ``LEROBOT_PORT`` (with a URDF path
via ``LEROBOT_URDF``) otherwise.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from a2a_robots import RobotAdapter, SafetyLimits, Workspace, serve_robot
from a2a_robots_lerobot import FakeLeRobot, LeRobotArmBackend, so101_arms


def build_adapter(
    *,
    fake: bool = False,
    port: str = "/dev/ttyACM0",
    urdf: str | Path | None = None,
    robot_id: str = "so101_follower",
) -> RobotAdapter:
    if fake:
        robot = FakeLeRobot(bimanual=False, dof=5)
        backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics)
        ee_pos = backend.observe()["right"].pose.position
        robot.place_object("cube", ee_pos)
    else:
        if urdf is None:
            raise ValueError("--urdf (or LEROBOT_URDF) is required for live hardware")
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cfg = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            disable_torque_on_disconnect=False,
        )
        robot = SO101Follower(cfg)
        backend = LeRobotArmBackend(
            robot,
            so101_arms(urdf),
            connect=True,
        )

    limits = SafetyLimits(
        workspace=Workspace("world", (-0.5, -0.5, -0.1), (0.6, 0.5, 0.6)),
        max_step_m=0.05,
        max_step_rad=0.25,
    )
    return RobotAdapter(
        backend,
        limits,
        name="LeRobot SO-101",
        description="SO-101 follower arm controlled via Hugging Face LeRobot.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake",
        action="store_true",
        default=os.environ.get("LEROBOT_FAKE") == "1",
        help="Use offline FakeLeRobot instead of connecting to serial hardware.",
    )
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("LEROBOT_PORT", "/dev/ttyACM0"),
        help="Serial port for SO101Follower.",
    )
    parser.add_argument(
        "--urdf",
        default=os.environ.get("LEROBOT_URDF"),
        help="Path to SO-101 URDF file (e.g. from huggingface.co/datasets/lerobot/robot-urdfs).",
    )
    parser.add_argument("--id", default=os.environ.get("LEROBOT_ID", "so101_follower"))
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()
    serve_robot(
        build_adapter(fake=args.fake, port=args.serial_port, urdf=args.urdf, robot_id=args.id),
        port=args.port,
    )
