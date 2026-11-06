"""ROS 2 のタスク管理のノード。通路の開通の依頼（/task/clear_corridor）を受け、各ロボットの協調のノードへ割り振る。

依頼を受けたら、シーンの全ロボットの協調のノード（/<ロボット>/clear_corridor）へ同じ目標を依頼し、
箱の状態（/team/object_state）から進捗（箱から退避領域の中心までの距離・箱の中心が退避領域に入ったまま続いた時間）を返す。
全ロボットが成功を返したら成功、いずれかが失敗を返したら失敗とし、残りのロボットの依頼を取り消す。
依頼の時刻から task_manager.timeout_s（シミュレーション時間）たっても結果がそろわなければ、時間切れとして取り消す。

起動例（launch から起動する）:
    go2-ros2-task-manager --config configs/team_rule_based.yaml \\
        --ros-args -r __node:=task_manager -p use_sim_time:=true
"""

import argparse
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from go2_interfaces.action import ClearCorridor
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.client import ClientGoalHandle
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.task import Future
from rclpy.utilities import remove_ros_args

from cooperative_go2.entities.command_expiry import ns_to_seconds, seconds_to_ns
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    TeamRunSettings,
    load_ros2_team_settings,
    load_team_run_settings,
)
from cooperative_go2.interface_adapters.ros2_messages import read_odometry, time_to_ns
from cooperative_go2.interface_adapters.ros2_topics import (
    CLEAR_CORRIDOR_ACTION,
    OBJECT_STATE_TOPIC,
    SENSOR_QOS,
    TASK_CLEAR_CORRIDOR_ACTION,
    robot_topic,
)

NODE_NAME = "task_manager"
# 各ロボットの結果と、依頼の取り消し・時間切れを確かめる実時間の間隔 [s]
POLL_INTERVAL_S = 0.01
# 協調のノードが依頼を受け付けるのを待つ実時間の上限 [s]
ROBOT_REPLY_TIMEOUT_S = 10.0
# 受信待ちの区切り [s]（実時間）。SIGINT を受けてから終わるまでの遅れの上限になる
SPIN_TIMEOUT_S = 0.1
EXIT_INVALID_INPUT = 2

logger = logging.getLogger(__name__)


@dataclass
class _RobotTask:
    name: str
    handle: ClientGoalHandle | None = None
    result: ClearCorridor.Result | None = None
    status: int = GoalStatus.STATUS_UNKNOWN
    done: threading.Event = field(default_factory=threading.Event)

    def succeeded(self) -> bool:
        """協調のノードが作業を終え、成功を返した。"""
        return self.done.is_set() and self.status == GoalStatus.STATUS_SUCCEEDED and bool(self.result.success)

    def failed(self) -> bool:
        """協調のノードの依頼が、成功以外（失敗の結果・取り消し・中断）で終わった。"""
        return self.done.is_set() and not self.succeeded()

    def failure_reason(self) -> str:
        """失敗の理由（協調のノードが返した理由、結果が無ければ依頼の状態の番号）。"""
        if self.result is not None and self.result.termination_reason:
            return self.result.termination_reason
        return f"STATUS_{self.status}"


class TaskManagerNode:
    """タスク管理のノード。依頼の処理と受信を複数のスレッドで並行して行い、箱の状態はロックで守る。"""

    def __init__(self, node: Node, run: TeamRunSettings) -> None:
        """ノードと作業の設定を受け取り、依頼の受け口・各ロボットへの依頼先・箱の状態の購読を作る。"""
        self._node = node
        self._timeout_ns = seconds_to_ns(run.settings.team.task_manager.timeout_s)
        self._robot_names = tuple(robot.name for robot in run.scene.scene.robots)
        self._lock = threading.Lock()
        self._box_stamp_ns: int | None = None
        self._box_xy_m: tuple[float, float] | None = None
        self._stopping = threading.Event()
        group = ReentrantCallbackGroup()
        self._robot_clients = {
            name: ActionClient(node, ClearCorridor, robot_topic(name, CLEAR_CORRIDOR_ACTION), callback_group=group)
            for name in self._robot_names
        }
        node.create_subscription(Odometry, OBJECT_STATE_TOPIC, self._on_box, SENSOR_QOS, callback_group=group)
        self._server = ActionServer(
            node, ClearCorridor, TASK_CLEAR_CORRIDOR_ACTION, execute_callback=self._execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT, cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )  # fmt: skip

    def stop(self) -> None:
        """ノードを終える前に呼ぶ。依頼を待っている処理を抜けさせる。"""
        self._stopping.set()

    def _on_box(self, message: Odometry) -> None:
        box = read_odometry(message)
        position = box.pose.position_m[0]
        with self._lock:
            self._box_stamp_ns = box.stamp_ns
            self._box_xy_m = (float(position[0]), float(position[1]))

    def _execute(self, goal_handle: ServerGoalHandle) -> ClearCorridor.Result:
        request = goal_handle.request
        start_ns = time_to_ns(request.header.stamp)
        tasks = [_RobotTask(name) for name in self._robot_names]
        for task in tasks:
            if not self._send(task, request):
                self._cancel_all(tasks)
                goal_handle.abort()
                return self._result(False, f"ROBOT_UNAVAILABLE:{task.name}", start_ns)
        in_goal_since_ns: int | None = None
        while True:
            if self._stopping.is_set():
                # タスク管理が先に終わっても、各ロボットが押し続けないよう依頼を取り消してから抜ける
                self._cancel_all(tasks)
                return self._result(False, "NODE_SHUTDOWN", start_ns)
            if goal_handle.is_cancel_requested:
                self._cancel_all(tasks)
                goal_handle.canceled()
                return self._result(False, "CANCELED", start_ns)
            failed = [task for task in tasks if task.failed()]
            if failed:
                self._cancel_all(tasks)
                goal_handle.succeed()
                reasons = ",".join(f"{task.name}:{task.failure_reason()}" for task in failed)
                return self._result(False, f"FAILED({reasons})", start_ns)
            if all(task.succeeded() for task in tasks):
                goal_handle.succeed()
                return self._result(True, "SUCCESS", start_ns)
            with self._lock:
                box_stamp_ns, box_xy = self._box_stamp_ns, self._box_xy_m
            if box_stamp_ns is not None and box_xy is not None and box_stamp_ns >= start_ns:
                if box_stamp_ns - start_ns > self._timeout_ns:
                    self._cancel_all(tasks)
                    goal_handle.succeed()
                    return self._result(False, "TIMEOUT", start_ns)
                in_goal_since_ns = self._publish_feedback(goal_handle, box_stamp_ns, box_xy, in_goal_since_ns)
            threading.Event().wait(POLL_INTERVAL_S)

    def _send(self, task: _RobotTask, request: ClearCorridor.Goal) -> bool:
        client = self._robot_clients[task.name]
        if not client.wait_for_server(timeout_sec=ROBOT_REPLY_TIMEOUT_S):
            logger.error("%s の協調のノードが見つからない", task.name)
            return False
        goal = ClearCorridor.Goal()
        goal.header = request.header
        goal.goal_position = request.goal_position
        goal.goal_radius_m = request.goal_radius_m
        accepted = threading.Event()
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda _future: accepted.set())
        if not accepted.wait(ROBOT_REPLY_TIMEOUT_S) or not future.result().accepted:
            logger.error("%s の協調のノードが依頼を受け付けなかった", task.name)
            return False
        task.handle = future.result()

        def on_result(result_future: Future) -> None:
            response = result_future.result()
            task.status = response.status
            task.result = response.result
            task.done.set()

        task.handle.get_result_async().add_done_callback(on_result)
        return True

    def _cancel_all(self, tasks: list[_RobotTask]) -> None:
        for task in tasks:
            if task.handle is not None and not task.done.is_set():
                task.handle.cancel_goal_async()

    def _publish_feedback(
        self, goal_handle: ServerGoalHandle, stamp_ns: int, box_xy: tuple[float, float], in_goal_since_ns: int | None
    ) -> int | None:
        request = goal_handle.request
        dx = box_xy[0] - request.goal_position.x
        dy = box_xy[1] - request.goal_position.y
        distance_m = (dx * dx + dy * dy) ** 0.5
        in_goal = distance_m <= request.goal_radius_m
        since_ns = (in_goal_since_ns if in_goal_since_ns is not None else stamp_ns) if in_goal else None
        feedback = ClearCorridor.Feedback()
        feedback.box_goal_distance_m = distance_m
        feedback.in_goal_duration_s = 0.0 if since_ns is None else ns_to_seconds(stamp_ns - since_ns)
        goal_handle.publish_feedback(feedback)
        return since_ns

    def _result(self, success: bool, reason: str, start_ns: int) -> ClearCorridor.Result:
        result = ClearCorridor.Result()
        result.success = success
        result.termination_reason = reason
        with self._lock:
            last_ns = start_ns if self._box_stamp_ns is None else max(self._box_stamp_ns, start_ns)
        result.elapsed_sim_s = ns_to_seconds(last_ns - start_ns)
        return result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 のタスク管理のノード（通路の開通の依頼を各ロボットへ割り振る）")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="ROS 2 の実行系の設定（configs/ros2_team*.yaml 形式）。作業の設定はその team_config から読む",
    )  # fmt: skip
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """タスク管理のノードを動かす。設定が不正なら非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        run = load_team_run_settings(Path(load_ros2_team_settings(args.config).team_config))
    except (FileNotFoundError, ConfigValidationError) as error:
        logger.error("タスク管理のノードの設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    # rclpy にシグナルを処理させない（協調のノードと同じ理由）。SIGINT は KeyboardInterrupt として受ける
    rclpy.init(args=raw_argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node(NODE_NAME)
    manager = TaskManagerNode(node, run)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while True:
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except KeyboardInterrupt:
        logger.info("SIGINT を受けたのでタスク管理のノードを終える")
    finally:
        manager.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
