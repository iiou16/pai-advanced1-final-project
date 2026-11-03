"""ROS 2 の高レベル方策ノード。Go2 と箱の状態から、学習済みの方策で速度指令を決める。

Genesis を読み込まず、物理ステップを進められない。目標は ClearCorridor のアクションで受け付ける。
指令は、目標を依頼された時刻（試行の始め）の状態から、高レベル方策の周期（high_level_period_s）ごとの状態で決める。
学習時と同じく、エピソードの始めの状態から一定の周期で判断するため。

起動例（launch から起動する。Namespace でロボットを分ける）:
    go2-ros2-coord --config configs/ros2_push_single.yaml --metrics-dir outputs/ros2_eval/<run>/nodes \\
        --ros-args -r __ns:=/go2_a -r __node:=coord_policy -p use_sim_time:=true
"""

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
import torch
from geometry_msgs.msg import TwistStamped
from go2_interfaces.action import ClearCorridor
from go2_interfaces.msg import ContactState
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from cooperative_go2.entities.command_expiry import ns_to_seconds, seconds_to_ns
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    Ros2PushSettings,
    load_push_training_settings,
    load_ros2_push_settings,
)
from cooperative_go2.interface_adapters.push_evaluation_adapters import PolicyConditionMismatchError, load_push_policy
from cooperative_go2.interface_adapters.ros2_metrics import LatencyStats, write_node_metrics
from cooperative_go2.interface_adapters.ros2_messages import (
    StampedContact,
    StampedPose,
    make_twist_stamped,
    read_contact_state,
    read_odometry,
    time_to_ns,
)
from cooperative_go2.interface_adapters.ros2_topics import (
    BASE_LINK_FRAME,
    BOX_CONTACT_TOPIC,
    CLEAR_CORRIDOR_ACTION,
    CMD_VEL_TOPIC,
    COMMAND_QOS,
    OBJECT_STATE_TOPIC,
    ODOM_TOPIC,
    SENSOR_QOS,
    robot_frame,
)
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.use_cases.push_policy_driver import PushPolicyDriver, PushPolicyInputs
from cooperative_go2.use_cases.stamp_aligner import AlignedValues, StampAligner

NODE_NAME = "coord_policy"
DEVICE = "cpu"
ROBOT_SOURCE = "odom"
BOX_SOURCE = "object_state"
CONTACT_SOURCE = "box_contact"
SOURCES = (ROBOT_SOURCE, BOX_SOURCE, CONTACT_SOURCE)
MAX_PENDING_STAMPS = 10
# 依頼の取り消しや差し替えを確かめる実時間の間隔 [s]
GOAL_POLL_INTERVAL_S = 0.01
METRICS_FILENAME = "coord_policy_metrics.json"
# 受信待ちの区切り [s]（実時間）。SIGINT を受けてから終わるまでの遅れの上限になる
SPIN_TIMEOUT_S = 0.1
EXIT_INVALID_INPUT = 2

logger = logging.getLogger(__name__)

StateValue = StampedPose | StampedContact


@dataclass
class _ActiveGoal:
    handle: ServerGoalHandle
    start_stamp_ns: int
    finished: threading.Event
    last_decision_ns: int | None = None
    # 次に判断する、試行の開始から数えた周期の番号。判断の時刻を開始時刻 + 番号 × 周期の格子にそろえる
    next_decision_index: int = 0
    succeeded: bool = False


class CoordPolicyNode:
    """高レベル方策のノード。受信とアクションの処理を複数のスレッドで並行して行い、方策の状態はロックで守る。"""

    def __init__(self, node: Node, driver: PushPolicyDriver, high_level_period_s: float) -> None:
        """ノード・推論器・高レベル方策の周期 [s] を受け取り、購読・配信・アクションを作る。"""
        self._node = node
        self._driver = driver
        self._period_ns = seconds_to_ns(high_level_period_s)
        self._frame = robot_frame(node.get_namespace(), BASE_LINK_FRAME)
        self._lock = threading.Lock()
        self._aligner: StampAligner[StateValue] = StampAligner(SOURCES, MAX_PENDING_STAMPS)
        self._latest: AlignedValues[StateValue] | None = None
        self._goal: _ActiveGoal | None = None
        self._stopping = threading.Event()
        self.inference = LatencyStats()
        group = ReentrantCallbackGroup()
        self._cmd_vel = node.create_publisher(TwistStamped, CMD_VEL_TOPIC, COMMAND_QOS)
        node.create_subscription(Odometry, ODOM_TOPIC, self._on_robot, SENSOR_QOS, callback_group=group)
        node.create_subscription(Odometry, OBJECT_STATE_TOPIC, self._on_box, SENSOR_QOS, callback_group=group)
        node.create_subscription(ContactState, BOX_CONTACT_TOPIC, self._on_contact, SENSOR_QOS, callback_group=group)
        self._server = ActionServer(
            node, ClearCorridor, CLEAR_CORRIDOR_ACTION, execute_callback=self._execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT, cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )  # fmt: skip

    def stop(self) -> None:
        """ノードを終える前に呼ぶ。依頼を待っている処理を抜けさせる。"""
        self._stopping.set()

    def _on_robot(self, message: Odometry) -> None:
        robot = read_odometry(message)
        self._put(ROBOT_SOURCE, robot.stamp_ns, robot)

    def _on_box(self, message: Odometry) -> None:
        box = read_odometry(message)
        self._put(BOX_SOURCE, box.stamp_ns, box)

    def _on_contact(self, message: ContactState) -> None:
        contact = read_contact_state(message)
        self._put(CONTACT_SOURCE, contact.stamp_ns, contact)

    def _put(self, source: str, stamp_ns: int, value: StateValue) -> None:
        with self._lock:
            self._aligner.put(source, stamp_ns, value)
            aligned = self._aligner.pop_latest_complete_or_none()
            if aligned is None:
                return
            self._latest = aligned
            self._decide_if_due(aligned)

    def _execute(self, goal_handle: ServerGoalHandle) -> ClearCorridor.Result:
        request = goal_handle.request
        goal = _ActiveGoal(goal_handle, time_to_ns(request.header.stamp), threading.Event())
        with self._lock:
            self._driver.start((request.goal_position.x, request.goal_position.y))
            self._goal = goal
            # 試行の始めの状態が依頼より先に届いていれば、その状態で最初の指令を決める
            if self._latest is not None:
                self._decide_if_due(self._latest)
        while not goal.finished.wait(GOAL_POLL_INTERVAL_S):
            if goal_handle.is_cancel_requested or self._goal is not goal:
                break
            if self._stopping.is_set():
                # ノードを終える（launch の終了や停止の確認）。依頼の結果は返せないので、待つのをやめて抜ける
                return ClearCorridor.Result(termination_reason="NODE_SHUTDOWN")
        with self._lock:
            if self._goal is goal:
                self._goal = None
        return self._finish(goal)

    def _finish(self, goal: _ActiveGoal) -> ClearCorridor.Result:
        result = ClearCorridor.Result()
        last = goal.start_stamp_ns if goal.last_decision_ns is None else goal.last_decision_ns
        result.elapsed_sim_s = ns_to_seconds(last - goal.start_stamp_ns)
        result.success = goal.succeeded
        if goal.succeeded:
            result.termination_reason = "SUCCESS"
            goal.handle.succeed()
        elif goal.handle.is_cancel_requested:
            result.termination_reason = "CANCELED"
            goal.handle.canceled()
        else:
            # 次の依頼に差し替えられた
            result.termination_reason = "PREEMPTED"
            goal.handle.abort()
        return result

    def _decide_if_due(self, aligned: AlignedValues[StateValue]) -> None:
        # 前提: self._lock を持っている
        goal = self._goal
        if goal is None or goal.finished.is_set() or aligned.stamp_ns < goal.start_stamp_ns:
            return
        # 状態を取りこぼして格子の時刻より後の状態で判断しても、次の判断は格子の時刻に戻す（判断の時刻がずれ続けない）
        elapsed_ns = aligned.stamp_ns - goal.start_stamp_ns
        if elapsed_ns < goal.next_decision_index * self._period_ns:
            return
        goal.next_decision_index = elapsed_ns // self._period_ns + 1
        robot, box, contact = (aligned.values[source] for source in SOURCES)
        if not isinstance(robot, StampedPose) or not isinstance(box, StampedPose):
            raise TypeError(f"組にした状態の型が想定と違う: {type(robot)}, {type(box)}")
        if not isinstance(contact, StampedContact):
            raise TypeError(f"組にした接触状態の型が想定と違う: {type(contact)}")
        inputs = PushPolicyInputs(
            robot=robot.pose.to_planar(),
            robot_projected_gravity=robot.pose.projected_gravity(),
            box=box.pose.to_planar(),
            box_contact=contact.in_contact,
        )
        started = time.perf_counter()
        decision = self._driver.decide(inputs)
        self.inference.add(time.perf_counter() - started)
        goal.last_decision_ns = aligned.stamp_ns
        if decision.succeeded:
            # 成功したら止まる。歩行方策のノードは停止の指令を受けて立ち止まる
            stop = torch.zeros((1, VELOCITY_COMMAND_SIZE))
            self._cmd_vel.publish(make_twist_stamped(aligned.stamp_ns, self._frame, stop))
            goal.succeeded = True
            goal.finished.set()
            return
        self._cmd_vel.publish(make_twist_stamped(aligned.stamp_ns, self._frame, decision.commands))
        feedback = ClearCorridor.Feedback()
        feedback.box_goal_distance_m = decision.box_goal_distance_m
        feedback.in_goal_duration_s = decision.in_goal_duration_s
        goal.handle.publish_feedback(feedback)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 の高レベル方策ノード")
    parser.add_argument(
        "--config", type=Path, required=True, help="ROS 2 の実行系の設定（configs/ros2_push_single.yaml 形式）"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, required=True, help="終了時に計測値（推論時間など）を書くディレクトリ"
    )
    return parser.parse_args(argv)


def _build_driver(settings: Ros2PushSettings) -> tuple[PushPolicyDriver, float]:
    push = load_push_training_settings(Path(settings.push_config))
    policy = load_push_policy(Path(settings.push_policy.run_dir), settings.push_policy.checkpoint, push, DEVICE)
    return PushPolicyDriver(policy, push.task), push.task.episode.high_level_period_s


def main(argv: list[str] | None = None) -> None:
    """高レベル方策のノードを動かす。終了時に計測値を書く。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        settings = load_ros2_push_settings(args.config)
        driver, period_s = _build_driver(settings)
    except (FileNotFoundError, ConfigValidationError, PolicyConditionMismatchError, CheckpointNotFoundError,
            CheckpointLoadError) as error:  # fmt: skip
        logger.error("高レベル方策のノードの設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    torch.set_num_threads(settings.runtime.torch_num_threads)
    # rclpy にシグナルを処理させない。rclpy が受信の途中でコンテキストを閉じると、受信の処理が内部エラーで落ちるため。
    # SIGINT は KeyboardInterrupt として受け、依頼を待つ処理と受信を止めてから閉じる
    rclpy.init(args=raw_argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node(NODE_NAME)
    coord_node = CoordPolicyNode(node, driver, period_s)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        # 受信待ちを短く区切る。区切らないと、メッセージが途絶えたときに KeyboardInterrupt が上がらない
        while True:
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except KeyboardInterrupt:
        # launch の終了や停止の確認（SIGINT）で止める。正常に終えたときだけ計測値を書く（異常終了の計測値を残さない）
        logger.info("SIGINT を受けたので高レベル方策のノードを終える")
        write_node_metrics(args.metrics_dir / METRICS_FILENAME, {"inference_s": coord_node.inference.summary()})
    finally:
        coord_node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
