from a2a_robots.testing import BackendContract
from a2a_robots_lerobot import FakeLeRobot, LeRobotArmBackend


class TestLeRobotSO101Contract(BackendContract):
    def make_backend(self) -> LeRobotArmBackend:
        robot = FakeLeRobot(bimanual=False, dof=5)
        return LeRobotArmBackend(
            robot,
            robot.arms,
            solvers=robot.kinematics,
            realtime=False,
        )


class TestLeRobotBiSOContract(BackendContract):
    def make_backend(self) -> LeRobotArmBackend:
        robot = FakeLeRobot(bimanual=True, dof=5)
        return LeRobotArmBackend(
            robot,
            robot.arms,
            solvers=robot.kinematics,
            realtime=False,
        )


class TestLeRobotSixDofContract(BackendContract):
    def make_backend(self) -> LeRobotArmBackend:
        robot = FakeLeRobot(bimanual=False, dof=6)
        return LeRobotArmBackend(
            robot,
            robot.arms,
            solvers=robot.kinematics,
            realtime=False,
        )
