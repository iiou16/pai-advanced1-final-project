"""通路の開通の試行の流れ（use_cases.ros_team_trial）が使う、ROS 2 とのやり取りの実装。

シミュレーションノードは全ロボットのトピックを 1 つのノードで扱うので、ロボットごとのトピックは絶対名
（例: /go2_a/odom）で作る。歩行方策・協調のノードは各ロボットの Namespace の相対名で同じトピックにつながる。
前提: ノードの受信の処理（executor）は、試行の流れとは別のスレッドで回っていること。
"""

import threading
import time

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Point
from go2_interfaces.action import ClearCorridor
from go2_interfaces.msg import AgentStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import JointTrajectory

from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.interface_adapters.ros2_messages import (
    make_clock,
    make_joint_state,
    make_odometry,
    make_transform,
    ns_to_time,
    read_joint_trajectory,
)
from cooperative_go2.interface_adapters.ros2_metrics import RoundTripLatencyProbe
from cooperative_go2.interface_adapters.ros2_team_messages import read_agent_status
from cooperative_go2.interface_adapters.ros2_topics import (
    AGENT_STATUS_TOPIC,
    BASE_LINK_FRAME,
    BOX_FRAME,
    CLOCK_QOS,
    CLOCK_TOPIC,
    CMD_VEL_TOPIC,
    COMMAND_QOS,
    GAIT_RESET_SERVICE,
    JOINT_STATES_TOPIC,
    JOINT_TARGETS_TOPIC,
    OBJECT_STATE_TOPIC,
    ODOM_TOPIC,
    SENSOR_QOS,
    TASK_CLEAR_CORRIDOR_ACTION,
    TEAM_STATUS_QOS,
    WORLD_FRAME,
    robot_topic,
)
from cooperative_go2.interface_adapters.ros2_sim_ports import PeerNodeRejectedError, PeerNodeTimeoutError
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler
from cooperative_go2.use_cases.ros_team_trial import TaskResult, TeamWorldState

# 方策のノードの配信が見えるかを確かめる実時間の間隔 [s]
PEER_DISCOVERY_POLL_S = 0.1
ACCEPTED_CANCEL_RETURN_CODES = frozenset({CancelGoal.Response.ERROR_NONE, CancelGoal.Response.ERROR_GOAL_TERMINATED})


class RosTeamStatePublisher:
    """時刻・全ロボットの関節と胴体・箱を、ロボットごとの Namespace のトピックと TF に配信する。"""

    def __init__(self, node: Node, robot_names: tuple[str, ...], latency_probe: RoundTripLatencyProbe) -> None:
        """配信に使うノード・ロボットの名前（状態の並びと同じ順）・往復の遅延を測るものを受け取る。

        往復の遅延は、先頭のロボットの関節の状態と関節目標で測る。
        """
        self._robot_names = robot_names
        self._latency_probe = latency_probe
        self._clock = node.create_publisher(Clock, CLOCK_TOPIC, CLOCK_QOS)
        self._joint_states = [
            node.create_publisher(JointState, robot_topic(name, JOINT_STATES_TOPIC), SENSOR_QOS)
            for name in robot_names
        ]
        self._odom = [
            node.create_publisher(Odometry, robot_topic(name, ODOM_TOPIC), SENSOR_QOS) for name in robot_names
        ]
        self._object_state = node.create_publisher(Odometry, OBJECT_STATE_TOPIC, SENSOR_QOS)
        self._base_frames = [f"{name}/{BASE_LINK_FRAME}" for name in robot_names]
        self._tf = TransformBroadcaster(node)

    def publish(self, stamp_ns: int, state: TeamWorldState) -> None:
        """時刻 stamp_ns [ns] の状態を配信する。/clock を先に配信する。"""
        self._clock.publish(make_clock(stamp_ns))
        transforms = []
        for index, (go2, robot) in enumerate(zip(state.go2, state.robots, strict=True)):
            proprioception = go2.proprioception
            joint_state = make_joint_state(stamp_ns, proprioception.dof_pos, proprioception.dof_vel)
            self._joint_states[index].publish(joint_state)
            if index == 0:
                self._latency_probe.on_published(stamp_ns)
            self._odom[index].publish(make_odometry(stamp_ns, WORLD_FRAME, self._base_frames[index], robot))
            transforms.append(make_transform(stamp_ns, WORLD_FRAME, self._base_frames[index], robot))
        self._object_state.publish(make_odometry(stamp_ns, WORLD_FRAME, BOX_FRAME, state.box))
        transforms.append(make_transform(stamp_ns, WORLD_FRAME, BOX_FRAME, state.box))
        self._tf.sendTransform(transforms)


def subscribe_team_joint_targets(
    node: Node, robot_names: tuple[str, ...], schedulers: tuple[JointTargetScheduler, ...],
    latency_probe: RoundTripLatencyProbe,
) -> None:  # fmt: skip
    """各ロボットの歩行方策のノードの関節目標を受け取り、そのロボットの scheduler に渡す購読を作る。

    ロボットのトピックごとに別の購読を作り、目標を別のロボットの scheduler に渡さない（観測と行動を混線させない）。
    """
    for index, (name, scheduler) in enumerate(zip(robot_names, schedulers, strict=True)):

        def on_joint_targets(message: JointTrajectory, scheduler: JointTargetScheduler = scheduler,
                             measure: bool = index == 0) -> None:  # fmt: skip
            targets = read_joint_trajectory(message)
            if measure:
                latency_probe.on_target_received(targets.stamp_ns)
            scheduler.receive(targets.stamp_ns, targets.joint_targets)

        topic = robot_topic(name, JOINT_TARGETS_TOPIC)
        node.create_subscription(JointTrajectory, topic, on_joint_targets, COMMAND_QOS)


class RosTeamTaskPeers:
    """全ロボットの歩行方策のノードへのリセットの依頼と、タスク管理への通路の開通の依頼を行う。

    作業の状態（/team/agent_status）を購読し、試行ごとに各ロボットが初めて PUSH を配信した時刻を記録する
    （押し始めの時刻のそろい方の評価に使う）。
    """

    def __init__(self, node: Node, robot_names: tuple[str, ...], reply_timeout_s: float) -> None:
        """依頼に使うノード・ロボットの名前・応答を待つ実時間の上限 [s] を受け取る。"""
        self._node = node
        self._robot_names = robot_names
        self._reply_timeout_s = reply_timeout_s
        self._reset_clients = [
            node.create_client(Trigger, robot_topic(name, GAIT_RESET_SERVICE)) for name in robot_names
        ]
        self._task_client = ActionClient(node, ClearCorridor, TASK_CLEAR_CORRIDOR_ACTION)
        self._lock = threading.Lock()
        self._goal_handle: ClientGoalHandle | None = None
        self._task_result: TaskResult | None = None
        self._trial_start_ns: int | None = None
        self._first_push_ns: dict[str, int] = {}
        node.create_subscription(AgentStatus, AGENT_STATUS_TOPIC, self._on_agent_status, TEAM_STATUS_QOS)

    def wait_for_peers(self, timeout_s: float) -> None:
        """各ロボットの歩行方策のリセットのサービス・タスク管理のアクション・各ロボットの指令の配信が見えるまで待つ。

        Raises:
            PeerNodeTimeoutError: timeout_s [s]（実時間）以内に使えるようにならなかった。
        """
        for name, client in zip(self._robot_names, self._reset_clients, strict=True):
            if not client.wait_for_service(timeout_sec=timeout_s):
                raise PeerNodeTimeoutError(f"{name} の歩行方策のサービスが {timeout_s} s 以内に現れない")
        if not self._task_client.wait_for_server(timeout_sec=timeout_s):
            raise PeerNodeTimeoutError(
                f"タスク管理のアクション {TASK_CLEAR_CORRIDOR_ACTION} が {timeout_s} s 以内に現れない"
            )
        deadline = time.monotonic() + timeout_s
        for name in self._robot_names:
            for topic in (robot_topic(name, JOINT_TARGETS_TOPIC), robot_topic(name, CMD_VEL_TOPIC)):
                while self._node.count_publishers(topic) == 0:
                    if time.monotonic() > deadline:
                        raise PeerNodeTimeoutError(f"トピック {topic} の配信が {timeout_s} s 以内に現れない")
                    time.sleep(PEER_DISCOVERY_POLL_S)

    def reset_gait_policies(self) -> None:
        """全ロボットの歩行方策のノードに前回の行動を 0 に戻させ、応答を待つ。

        Raises:
            PeerNodeTimeoutError: 時間内に応答が無い。
            PeerNodeRejectedError: 歩行方策のノードがリセットに失敗したと応えた。
        """
        for name, client in zip(self._robot_names, self._reset_clients, strict=True):
            response = self._wait(client.call_async(Trigger.Request()), f"{name} の歩行方策のリセット")
            if not response.success:
                raise PeerNodeRejectedError(f"{name} の歩行方策のノードがリセットに失敗した: {response.message}")

    def start_task(self, start_stamp_ns: int, retreat_xy_m: tuple[float, float], retreat_radius_m: float) -> None:
        """タスク管理に、時刻 start_stamp_ns [ns] に始めた試行の通路の開通を依頼し、受け付けられるまで待つ。

        Raises:
            PeerNodeTimeoutError: 時間内に応答が無い。
            PeerNodeRejectedError: タスク管理が依頼を断った。
        """
        with self._lock:
            self._trial_start_ns = start_stamp_ns
            self._first_push_ns = {}
            self._task_result = None
        goal = ClearCorridor.Goal()
        goal.header.stamp = ns_to_time(start_stamp_ns)
        goal.header.frame_id = WORLD_FRAME
        goal.goal_position = Point(x=retreat_xy_m[0], y=retreat_xy_m[1], z=0.0)
        goal.goal_radius_m = retreat_radius_m
        handle = self._wait(self._task_client.send_goal_async(goal), "通路の開通の依頼")
        if not handle.accepted:
            raise PeerNodeRejectedError("タスク管理が通路の開通の依頼を断った")
        self._goal_handle = handle

        def on_result(future: Future) -> None:
            response = future.result()
            # 前の試行の依頼の結果が遅れて届いても、今の依頼の結果として扱わない
            if self._goal_handle is not handle or response.status != GoalStatus.STATUS_SUCCEEDED:
                return
            with self._lock:
                self._task_result = TaskResult(
                    success=bool(response.result.success), termination_reason=response.result.termination_reason
                )

        handle.get_result_async().add_done_callback(on_result)

    def find_task_result_or_none(self) -> TaskResult | None:
        """今の依頼の結果が届いていれば返す。まだなら None。"""
        with self._lock:
            return self._task_result

    def find_first_push_stamps_ns(self) -> dict[str, int]:
        """今の試行で、各ロボットが初めて PUSH を配信した時刻 [ns]。まだ押していないロボットは含めない。"""
        with self._lock:
            return dict(self._first_push_ns)

    def end_task(self) -> None:
        """依頼がまだ終わっていなければ取り消し、取り消しの応答を待つ。

        Raises:
            PeerNodeTimeoutError: タスク管理が居るのに、時間内に応答が無い。
            PeerNodeRejectedError: 取り消しを断られた。
        """
        handle = self._goal_handle
        self._goal_handle = None
        if handle is None or self.find_task_result_or_none() is not None:
            return
        response = self._wait(handle.cancel_goal_async(), "通路の開通の依頼の取り消し")
        if response.return_code not in ACCEPTED_CANCEL_RETURN_CODES:
            raise PeerNodeRejectedError(f"タスク管理が依頼の取り消しを断った（return_code={response.return_code}）")

    def _on_agent_status(self, message: AgentStatus) -> None:
        report = read_agent_status(message)
        with self._lock:
            start_ns = self._trial_start_ns
            if start_ns is None or report.stamp_ns < start_ns or report.phase is not TeamPhase.PUSH:
                return
            self._first_push_ns.setdefault(report.agent_id, report.stamp_ns)

    def _wait(self, future: Future, what: str) -> object:
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(self._reply_timeout_s):
            raise PeerNodeTimeoutError(f"{what}に {self._reply_timeout_s} s 以内に応答が無い")
        return future.result()
