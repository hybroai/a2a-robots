"""Serve turtlesim through the ROS 2 bridge on http://127.0.0.1:9200/.

Start turtlesim first, in a shell with ROS 2 sourced:

    ros2 run turtlesim turtlesim_node
"""

from a2a_robots_ros2 import Ros2BridgeAdapter, serve_bridge

SKILLS = [
    {
        "id": "rotate",
        "kind": "action",
        "ros_name": "/turtle1/rotate_absolute",
        "type": "turtlesim/action/RotateAbsolute",
        "description": "Turn the turtle to an absolute heading, theta in radians.",
        "timeout_s": 30,
        "exclusive": True,
    },
    {
        "id": "teleport",
        "kind": "service",
        "ros_name": "/turtle1/teleport_absolute",
        "type": "turtlesim/srv/TeleportAbsolute",
        "description": "Move the turtle to x, y (0 to 11) and heading theta.",
    },
    {
        "id": "spawn",
        "kind": "service",
        "ros_name": "/spawn",
        "type": "turtlesim/srv/Spawn",
        "description": "Add a turtle at x, y, theta; returns its name.",
    },
    {
        "id": "pose",
        "kind": "topic",
        "ros_name": "/turtle1/pose",
        "type": "turtlesim/msg/Pose",
        "description": "Where turtle1 is now.",
    },
]


def make_adapter() -> Ros2BridgeAdapter:
    from a2a_robots_ros2.rclpy_gateway import RclpyGateway

    return Ros2BridgeAdapter(
        SKILLS,
        RclpyGateway("turtlesim_bridge"),
        name="Turtlesim",
        description="turtle1 in turtlesim, through the allowlisted ROS 2 bridge.",
    )


if __name__ == "__main__":
    serve_bridge(make_adapter())
