"""ROS 2 の歩行方策ノード。関節の状態と速度指令から、12 関節の目標角を出す。

Genesis を読み込まず、物理ステップを進められない。受け取った状態ごとに 1 回推論し、目標を配信する。
速度指令が有効期限より古ければ停止の指令（すべて 0）で歩かせ、高レベル方策のノードが止まったらその場で立ち止まる。

起動例（launch から起動する。Namespace でロボットを分ける）:
    go2-ros2-gait --config configs/ros2_push_single.yaml \\
        --ros-args -r __ns:=/go2_a -r __node:=gait_policy -p use_sim_time:=true
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import rclpy
import torch
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from cooperative_go2.entities.command_expiry import ns_to_seconds, seconds_to_ns
from cooperative_go2.entities.rigid_body_pose import proprioception_from_pose
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    Ros2PushSettings,
    load_push_training_settings,
    load_ros2_push_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.ros2_messages import (
    StampedJointState,
    StampedPose,
    make_joint_trajectory,
    read_joint_state,
    read_odometry,
    read_twist_stamped,
)
from cooperative_go2.interface_adapters.ros2_metrics import LatencyStats, write_node_metrics
from cooperative_go2.interface_adapters.ros2_topics import (
    CMD_VEL_TOPIC,
    COMMAND_QOS,
    GAIT_RESET_SERVICE,
    JOINT_STATES_TOPIC,
    JOINT_TARGETS_TOPIC,
    ODOM_TOPIC,
    SENSOR_QOS,
)
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.use_cases.gait_command_follower import GaitCommandFollower, StampedVelocityCommand
from cooperative_go2.use_cases.stamp_aligner import StampAligner

NODE_NAME = "gait_policy"
DEVICE = "cpu"
JOINT_SOURCE = "joint_states"
ODOM_SOURCE = "odom"
# そろうのを待つ時刻の数。シミュレーションは 1 物理ステップごとに配信するので、数ステップ分あれば足りる
MAX_PENDING_STAMPS = 10
METRICS_FILENAME = "gait_policy_metrics.json"
# 受信待ちの区切り [s]（実時間）。SIGINT を受けてから終わるまでの遅れの上限になる
SPIN_TIMEOUT_S = 0.1
EXIT_INVALID_INPUT = 2

logger = logging.getLogger(__name__)


class GaitPolicyNode:
    """歩行方策のノード。受信のコールバックは 1 つのスレッドで順に処理する（SingleThreadedExecutor）。"""

    def __init__(self, node: Node, follower: GaitCommandFollower) -> None:
        """ノードと推論器を受け取り、購読・配信・リセットのサービスを作る。"""
        self._node = node
        self._follower = follower
        self._aligner: StampAligner[StampedJointState | StampedPose] = StampAligner(
            (JOINT_SOURCE, ODOM_SOURCE), MAX_PENDING_STAMPS
        )
        self._command: StampedVelocityCommand | None = None
        # 今の試行の最初の状態の時刻。これより前の時刻の速度指令は前の試行のものなので使わない
        self._episode_start_ns: int | None = None
        self.inference = LatencyStats()
        # 状態の時刻から見た、使った速度指令の古さ（シミュレーション時間）。速度指令の有効期限を決める根拠にする
        self.command_age = LatencyStats()
        self.stopped_steps = 0
        self._targets = node.create_publisher(JointTrajectory, JOINT_TARGETS_TOPIC, COMMAND_QOS)
        node.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_state, SENSOR_QOS)
        node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, SENSOR_QOS)
        node.create_subscription(TwistStamped, CMD_VEL_TOPIC, self._on_cmd_vel, COMMAND_QOS)
        node.create_service(Trigger, GAIT_RESET_SERVICE, self._on_reset)

    def _on_cmd_vel(self, message: TwistStamped) -> None:
        self._command = read_twist_stamped(message)

    def _on_joint_state(self, message: JointState) -> None:
        state = read_joint_state(message)
        self._aligner.put(JOINT_SOURCE, state.stamp_ns, state)
        self._step_if_aligned()

    def _on_odom(self, message: Odometry) -> None:
        odom = read_odometry(message)
        self._aligner.put(ODOM_SOURCE, odom.stamp_ns, odom)
        self._step_if_aligned()

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        # ロボットを初期姿勢に戻した。前回の行動と、前のエピソードの状態・指令を捨てる
        self._follower.reset()
        self._aligner = StampAligner((JOINT_SOURCE, ODOM_SOURCE), MAX_PENDING_STAMPS)
        self._command = None
        self._episode_start_ns = None
        response.success = True
        return response

    def _step_if_aligned(self) -> None:
        aligned = self._aligner.pop_latest_complete_or_none()
        if aligned is None:
            return
        joint_state = aligned.values[JOINT_SOURCE]
        odom = aligned.values[ODOM_SOURCE]
        if not isinstance(joint_state, StampedJointState) or not isinstance(odom, StampedPose):
            raise TypeError(f"組にした状態の型が想定と違う: {type(joint_state)}, {type(odom)}")
        proprioception = proprioception_from_pose(odom.pose, joint_state.dof_pos, joint_state.dof_vel)
        if self._episode_start_ns is None:
            self._episode_start_ns = aligned.stamp_ns
        command = self._command
        if command is not None and command.stamp_ns < self._episode_start_ns:
            # リセットの後に届いた、前の試行の状態から決めた指令
            command = None
        if command is not None:
            self.command_age.add(ns_to_seconds(aligned.stamp_ns - command.stamp_ns))
        started = time.perf_counter()
        output = self._follower.compute(proprioception, aligned.stamp_ns, command)
        self.inference.add(time.perf_counter() - started)
        if output.command_was_stopped:
            self.stopped_steps += 1
        self._targets.publish(make_joint_trajectory(aligned.stamp_ns, output.joint_targets))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 の歩行方策ノード")
    parser.add_argument(
        "--config", type=Path, required=True, help="ROS 2 の実行系の設定（configs/ros2_push_single.yaml 形式）"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, required=True, help="終了時に計測値（推論時間など）を書くディレクトリ"
    )
    return parser.parse_args(argv)


def _build_follower(settings: Ros2PushSettings) -> GaitCommandFollower:
    gait_policy = load_push_training_settings(Path(settings.push_config)).gait_policy
    controller = load_gait_controller(Path(gait_policy.run_dir), gait_policy.checkpoint, 1, DEVICE)
    return GaitCommandFollower(controller, seconds_to_ns(settings.command_expiry.velocity_command_s))


def main(argv: list[str] | None = None) -> None:
    """歩行方策のノードを動かす。終了時に計測値を書く。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        settings = load_ros2_push_settings(args.config)
        follower = _build_follower(settings)
    except (FileNotFoundError, ConfigValidationError, CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("歩行方策のノードの設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    torch.set_num_threads(settings.runtime.torch_num_threads)
    # rclpy にシグナルを処理させない。rclpy が受信の途中でコンテキストを閉じると、受信の処理が内部エラーで落ちるため。
    # SIGINT は KeyboardInterrupt として受け、受信を止めてから閉じる
    rclpy.init(args=raw_argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node(NODE_NAME)
    gait_node = GaitPolicyNode(node, follower)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # 受信待ちを短く区切る。区切らないと、メッセージが途絶えたときに KeyboardInterrupt が上がらない
        while True:
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except KeyboardInterrupt:
        # launch の終了や停止の確認（SIGINT）で止める。正常に終えたときだけ計測値を書く（異常終了の計測値を残さない）
        logger.info("SIGINT を受けたので歩行方策のノードを終える")
        write_node_metrics(
            args.metrics_dir / METRICS_FILENAME,
            {
                "inference_s": gait_node.inference.summary(),
                "velocity_command_age_sim_s": gait_node.command_age.summary(),
                "stopped_command_steps": gait_node.stopped_steps,
            },
        )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
