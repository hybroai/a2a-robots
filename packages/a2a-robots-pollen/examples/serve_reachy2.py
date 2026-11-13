"""Serve a Pollen Robotics Reachy 2 robot on http://127.0.0.1:9100/.

Uses ``FakeReachySDK`` when ``--fake`` is passed or ``REACHY2_FAKE=1`` is set,
and connects to a live or simulated Reachy 2 gRPC server at ``REACHY2_HOST``
(default ``"localhost"``) otherwise.
"""

from __future__ import annotations

import argparse
import os

from a2a_robots import RobotAdapter, SafetyLimits, Workspace, serve_robot
from a2a_robots_pollen import FakeReachySDK, Reachy2Backend, connect


def build_adapter(*, fake: bool = False, host: str = "localhost") -> RobotAdapter:
    if fake:
        reachy = FakeReachySDK()
        assert reachy.r_arm is not None
        pos = reachy.r_arm.forward_kinematics(
            reachy.r_arm.get_current_positions(degrees=False), degrees=False
        )[:3, 3]
        reachy.place_object("cube", (float(pos[0]), float(pos[1]), float(pos[2])))
    else:
        reachy = connect(host)

    backend = Reachy2Backend(reachy)
    limits = SafetyLimits(
        workspace=Workspace("world", (-0.5, -0.8, -0.5), (0.8, 0.8, 0.8)),
        max_step_m=0.05,
        max_step_rad=0.25,
    )
    return RobotAdapter(
        backend,
        limits,
        name="Pollen Reachy 2",
        description="Pollen Robotics Reachy 2 dual-arm humanoid robot via reachy2-sdk.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake",
        action="store_true",
        default=os.environ.get("REACHY2_FAKE") == "1",
        help="Use offline FakeReachySDK instead of connecting via gRPC.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("REACHY2_HOST", "localhost"),
        help="Reachy 2 gRPC host (default: localhost or REACHY2_HOST).",
    )
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()
    serve_robot(build_adapter(fake=args.fake, host=args.host), port=args.port)
