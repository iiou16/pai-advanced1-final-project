"""2 台の構成で、ロボットごとの観測（状態の配信）と行動（関節目標の受信）が混線しないことの検証。

シミュレーションノードが使う配信と受信の部品を、同じプロセスの別のノードと ROS 2 の通信でつないで確かめる。
"""

import threading
import time
from collections.abc import Callable, Iterator

import pytest
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.interface_adapters.ros2_messages import make_joint_trajectory, read_odometry
from cooperative_go2.interface_adapters.ros2_metrics import RoundTripLatencyProbe
from cooperative_go2.interface_adapters.ros2_team_sim_ports import RosTeamStatePublisher, subscribe_team_joint_targets
from cooperative_go2.interface_adapters.ros2_topics import COMMAND_QOS, SENSOR_QOS
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, JointTargetTiming, TargetSelection
from cooperative_go2.use_cases.ros_team_trial import TeamWorldState

ROBOTS = ("go2_a", "go2_b")
CONTROL_DT_NS = 20_000_000
STAMP_NS = 1_000_000_000
WAIT_S = 5.0


@pytest.fixture
def nodes() -> Iterator[tuple[Node, Node]]:
    """シミュレーションノード役と、方策・観測の相手役の 2 つのノード。受信は別のスレッドで回す。"""
    rclpy.init()
    sim = Node("isolation_sim")
    peer = Node("isolation_peer")
    executor = SingleThreadedExecutor()
    executor.add_node(sim)
    executor.add_node(peer)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield sim, peer
    executor.shutdown()
    sim.destroy_node()
    peer.destroy_node()
    rclpy.try_shutdown()


def wait_until(predicate: Callable[[], bool], publish: Callable[[], None]) -> None:
    """接続ができるまで配信を繰り返し、predicate が真になるまで待つ。"""
    deadline = time.monotonic() + WAIT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"{WAIT_S} s 以内に届かなかった")
        publish()
        time.sleep(0.05)


def schedulers() -> tuple[JointTargetScheduler, ...]:
    timing = JointTargetTiming(control_dt_ns=CONTROL_DT_NS, latency_steps=0, expiry_ns=5 * CONTROL_DT_NS)
    created = tuple(JointTargetScheduler(timing, torch.zeros(1, 12)) for _ in ROBOTS)
    for scheduler in created:
        scheduler.start_episode(0)
    return created


def test_joint_targets_reach_only_their_robot(nodes: tuple[Node, Node]) -> None:
    sim, peer = nodes
    robot_schedulers = schedulers()
    subscribe_team_joint_targets(sim, ROBOTS, robot_schedulers, RoundTripLatencyProbe())
    publisher = peer.create_publisher(JointTrajectory, "/go2_b/joint_targets", COMMAND_QOS)
    targets = torch.full((1, 12), 0.5)

    wait_until(
        lambda: robot_schedulers[1].select(STAMP_NS).selection is TargetSelection.POLICY,
        lambda: publisher.publish(make_joint_trajectory(STAMP_NS, targets)),
    )

    selected_b = robot_schedulers[1].select(STAMP_NS)
    selected_a = robot_schedulers[0].select(STAMP_NS)
    assert torch.equal(selected_b.joint_targets, targets)
    assert selected_a.selection is not TargetSelection.POLICY


def go2_state() -> Go2State:
    zeros = torch.zeros(1, 12)
    return Go2State(
        proprioception=Go2Proprioception(
            base_ang_vel=torch.zeros(1, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            dof_pos=zeros,
            dof_vel=zeros,
        ),
        base_pos_m=torch.zeros(1, 3),
        base_euler_deg=torch.zeros(1, 3),
        base_lin_vel=torch.zeros(1, 3),
    )


def pose_at(x_m: float, y_m: float) -> RigidBodyPose:
    return RigidBodyPose(
        position_m=torch.tensor([[x_m, y_m, 0.3]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_body_mps=torch.zeros(1, 3),
        ang_vel_body_radps=torch.zeros(1, 3),
    )


def test_each_robot_odometry_carries_its_own_state(nodes: tuple[Node, Node]) -> None:
    sim, peer = nodes
    publisher = RosTeamStatePublisher(sim, ROBOTS, RoundTripLatencyProbe())
    received: dict[str, Odometry] = {}

    def subscribe(robot: str) -> None:
        peer.create_subscription(Odometry, f"/{robot}/odom", lambda message: received.setdefault(robot, message),
                                 SENSOR_QOS)  # fmt: skip

    for robot in ROBOTS:
        subscribe(robot)
    state = TeamWorldState(
        go2=(go2_state(), go2_state()), robots=(pose_at(0.7, 0.25), pose_at(0.7, -0.25)), box=pose_at(1.4, 0.0)
    )

    wait_until(lambda: len(received) == len(ROBOTS), lambda: publisher.publish(STAMP_NS, state))

    a, b = (read_odometry(received[robot]) for robot in ROBOTS)
    assert float(a.pose.position_m[0, 1]) == pytest.approx(0.25)
    assert float(b.pose.position_m[0, 1]) == pytest.approx(-0.25)
    assert received["go2_a"].child_frame_id == "go2_a/base_link"
    assert received["go2_b"].child_frame_id == "go2_b/base_link"
