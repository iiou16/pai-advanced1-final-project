"""シミュレーションノードの試行の流れ（use_cases.ros_push_trial）が使う、ROS 2 とのやり取りの実装。

前提: ノードの受信の処理（executor）は、試行の流れとは別のスレッドで回っていること。
試行の流れのスレッドは、サービスとアクションの応答をここで待つ。
"""

import threading
import time

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Point
from go2_interfaces.action import ClearCorridor
from go2_interfaces.msg import ContactState
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

from cooperative_go2.interface_adapters.ros2_messages import (
    make_clock,
    make_contact_state,
    make_joint_state,
    make_odometry,
    make_transform,
    ns_to_time,
    read_joint_trajectory,
)
from cooperative_go2.interface_adapters.ros2_metrics import RoundTripLatencyProbe
from cooperative_go2.interface_adapters.ros2_topics import (
    BASE_LINK_FRAME,
    BOX_CONTACT_TOPIC,
    BOX_FRAME,
    CLEAR_CORRIDOR_ACTION,
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
    WORLD_FRAME,
    robot_frame,
)
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler
from cooperative_go2.use_cases.ros_push_trial import PushWorldState

# 方策のノードの配信が見えるかを確かめる実時間の間隔 [s]
PEER_DISCOVERY_POLL_S = 0.1
# 取り消しの応答のうち、依頼が残っていないことを表すもの（取り消した・取り消す前に終わっていた）
ACCEPTED_CANCEL_RETURN_CODES = frozenset({CancelGoal.Response.ERROR_NONE, CancelGoal.Response.ERROR_GOAL_TERMINATED})
TERMINAL_GOAL_STATUSES = frozenset({GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED})


class PeerNodeTimeoutError(RuntimeError):
    """方策のノードが、立ち上がらないか、リセットや目標の依頼に時間内に応えなかったことを表す。"""


class PeerNodeRejectedError(RuntimeError):
    """方策のノードが、リセットや目標の依頼を断ったことを表す。"""


class RosWorldStatePublisher:
    """時刻・Go2 の関節と胴体・箱・接触を、ロボットの Namespace のトピックと TF に配信する。"""

    def __init__(self, node: Node, latency_probe: RoundTripLatencyProbe) -> None:
        """配信に使うノードと、往復の遅延を測るものを受け取る。

        ノードの Namespace をロボットの名前として座標系の名前に付ける。
        """
        self._latency_probe = latency_probe
        namespace = node.get_namespace()
        self._base_frame = robot_frame(namespace, BASE_LINK_FRAME)
        self._clock = node.create_publisher(Clock, CLOCK_TOPIC, CLOCK_QOS)
        self._joint_states = node.create_publisher(JointState, JOINT_STATES_TOPIC, SENSOR_QOS)
        self._odom = node.create_publisher(Odometry, ODOM_TOPIC, SENSOR_QOS)
        self._object_state = node.create_publisher(Odometry, OBJECT_STATE_TOPIC, SENSOR_QOS)
        self._contact = node.create_publisher(ContactState, BOX_CONTACT_TOPIC, SENSOR_QOS)
        self._tf = TransformBroadcaster(node)

    def publish(self, stamp_ns: int, state: PushWorldState) -> None:
        """時刻 stamp_ns [ns] の状態を配信する。/clock を先に配信する。"""
        self._clock.publish(make_clock(stamp_ns))
        proprioception = state.go2.proprioception
        self._joint_states.publish(make_joint_state(stamp_ns, proprioception.dof_pos, proprioception.dof_vel))
        self._latency_probe.on_published(stamp_ns)
        self._odom.publish(make_odometry(stamp_ns, WORLD_FRAME, self._base_frame, state.robot))
        self._object_state.publish(make_odometry(stamp_ns, WORLD_FRAME, BOX_FRAME, state.box))
        self._contact.publish(make_contact_state(stamp_ns, state.box_contact))
        self._tf.sendTransform(
            [
                make_transform(stamp_ns, WORLD_FRAME, self._base_frame, state.robot),
                make_transform(stamp_ns, WORLD_FRAME, BOX_FRAME, state.box),
            ]
        )


def subscribe_joint_targets(node: Node, scheduler: JointTargetScheduler, latency_probe: RoundTripLatencyProbe) -> None:
    """歩行方策のノードの関節目標を受け取り、scheduler に渡す購読を作る。届いた時刻を latency_probe に伝える。"""

    def on_joint_targets(message: JointTrajectory) -> None:
        targets = read_joint_trajectory(message)
        latency_probe.on_target_received(targets.stamp_ns)
        scheduler.receive(targets.stamp_ns, targets.joint_targets)

    node.create_subscription(JointTrajectory, JOINT_TARGETS_TOPIC, on_joint_targets, COMMAND_QOS)


class RosPushTaskPeers:
    """歩行方策のノードへのリセットの依頼と、高レベル方策のノードへの目標の依頼を行う。"""

    def __init__(self, node: Node, reply_timeout_s: float) -> None:
        """依頼に使うノードと、応答を待つ実時間の上限 [s] を受け取る。"""
        self._node = node
        self._reply_timeout_s = reply_timeout_s
        self._reset_client = node.create_client(Trigger, GAIT_RESET_SERVICE)
        self._task_client = ActionClient(node, ClearCorridor, CLEAR_CORRIDOR_ACTION)
        self._goal_handle: ClientGoalHandle | None = None
        self._goal_status = GoalStatus.STATUS_UNKNOWN

    def wait_for_peers(self, timeout_s: float) -> None:
        """歩行方策のノードのリセットのサービス・高レベル方策のノードのアクション・両ノードの指令の配信が見えるまで待つ。

        Raises:
            PeerNodeTimeoutError: timeout_s [s]（実時間）以内に使えるようにならなかった。
        """
        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            raise PeerNodeTimeoutError(f"歩行方策のサービス {GAIT_RESET_SERVICE} が {timeout_s} s 以内に現れない")
        if not self._task_client.wait_for_server(timeout_sec=timeout_s):
            raise PeerNodeTimeoutError(f"高レベル方策のアクション {CLEAR_CORRIDOR_ACTION} が {timeout_s} s 以内に現れない")
        # 最初の試行の始めから指令と目標を受け取れるよう、方策のノードの配信が見えるまで待つ
        deadline = time.monotonic() + timeout_s
        for topic in (JOINT_TARGETS_TOPIC, CMD_VEL_TOPIC):
            while self._node.count_publishers(topic) == 0:
                if time.monotonic() > deadline:
                    raise PeerNodeTimeoutError(f"トピック {topic} の配信が {timeout_s} s 以内に現れない")
                time.sleep(PEER_DISCOVERY_POLL_S)

    def reset_gait_policy(self) -> None:
        """歩行方策のノードに前回の行動を 0 に戻させ、応答を待つ。

        Raises:
            PeerNodeTimeoutError: 時間内に応答が無い。
            PeerNodeRejectedError: 歩行方策のノードがリセットに失敗したと応えた。
        """
        response = self._wait(self._reset_client.call_async(Trigger.Request()), "歩行方策のリセット")
        if not response.success:
            raise PeerNodeRejectedError(f"歩行方策のノードがリセットに失敗した: {response.message}")

    def start_task(self, start_stamp_ns: int, goal_xy_m: tuple[float, float], goal_radius_m: float) -> None:
        """高レベル方策のノードに、時刻 start_stamp_ns [ns] に始めた試行の目標領域を依頼し、受け付けられるまで待つ。

        Raises:
            PeerNodeTimeoutError: 時間内に応答が無い。
            PeerNodeRejectedError: 高レベル方策のノードが依頼を断った。
        """
        goal = ClearCorridor.Goal()
        goal.header.stamp = ns_to_time(start_stamp_ns)
        goal.header.frame_id = WORLD_FRAME
        goal.goal_position = Point(x=goal_xy_m[0], y=goal_xy_m[1], z=0.0)
        goal.goal_radius_m = goal_radius_m
        handle = self._wait(self._task_client.send_goal_async(goal), "目標の依頼")
        if not handle.accepted:
            raise PeerNodeRejectedError("高レベル方策のノードが目標の依頼を断った")
        self._goal_handle = handle
        self._goal_status = GoalStatus.STATUS_ACCEPTED

        def on_result(future: Future) -> None:
            # 前の試行の依頼の結果が遅れて届いても、今の依頼の状態を書き換えない
            if self._goal_handle is handle:
                self._goal_status = future.result().status

        handle.get_result_async().add_done_callback(on_result)

    def end_task(self) -> None:
        """依頼がまだ終わっていなければ取り消し、取り消しの応答を待つ。

        高レベル方策のノードが既に居なければ取り消さない（居なくなったサーバに、取り消すべき依頼は残っていない）。
        方策のノードを止めて止まることを確かめる試行でも、試行の記録を残すため。

        Raises:
            PeerNodeTimeoutError: 高レベル方策のノードが居るのに、時間内に応答が無い。
            PeerNodeRejectedError: 依頼が試行の途中で中断（ABORTED）された、または取り消しを断られた。
                高レベル方策のノードの処理が失敗して指令が止まると、試行の結果が「止まった」ことと区別できなくなるため。
        """
        handle = self._goal_handle
        self._goal_handle = None
        if handle is None:
            return
        if self._goal_status == GoalStatus.STATUS_ABORTED:
            raise PeerNodeRejectedError("高レベル方策のノードの依頼が試行の途中で中断（ABORTED）された")
        if self._goal_status in TERMINAL_GOAL_STATUSES:
            return
        if not self._task_client.server_is_ready():
            self._node.get_logger().warning("高レベル方策のノードが居ないため、目標を取り消さずに試行を終える")
            return
        response = self._wait(handle.cancel_goal_async(), "目標の取り消し")
        if response.return_code not in ACCEPTED_CANCEL_RETURN_CODES:
            raise PeerNodeRejectedError(f"高レベル方策のノードが目標の取り消しを断った（return_code={response.return_code}）")

    def _wait(self, future: Future, what: str) -> object:
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(self._reply_timeout_s):
            raise PeerNodeTimeoutError(f"{what}に {self._reply_timeout_s} s 以内に応答が無い")
        return future.result()
