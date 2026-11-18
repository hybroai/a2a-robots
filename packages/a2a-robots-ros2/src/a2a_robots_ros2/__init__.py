"""An allowlisted ROS 2 bridge for a2a-robots: bring-up and debugging over A2A."""

__version__ = "0.1.2"

from .adapter import Ros2BridgeAdapter, serve_bridge
from .config import Kind, SkillSpec, parse_skills
from .gateway import GoalHandle, GoalResult, RequestError, RosError, RosGateway, TopicSample

__all__ = [
    "GoalHandle",
    "GoalResult",
    "Kind",
    "RequestError",
    "Ros2BridgeAdapter",
    "RosError",
    "RosGateway",
    "SkillSpec",
    "TopicSample",
    "__version__",
    "parse_skills",
    "serve_bridge",
]
