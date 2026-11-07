"""ROS 2 の協調のノード。1 台のロボットの、箱を押す判断（規則ベースの協調、または学習した協調の方策）を動かす。

Genesis を読み込まず、物理ステップを進められない。作業はタスク管理のノードから ClearCorridor のアクションで受ける。
依頼の時刻（試行の始め）から判断の周期ごとの状態で判断し、速度指令を決める。判断の中身は判断器（TeamDecider）に任せる。
作業の状態・押す意図・自分の位置と指令を /team/agent_status に配信する。協調のモード（cooperative・cooperative_wait）と
学習した方策（learned）では相手の状態を受け取って判断に使う。通信なしの独立制御（independent）と 1 台では相手の状態を
購読しない（配信は記録のために行う）。

起動例（launch から起動する。Namespace でロボットを分ける）:
    go2-ros2-team-coord --config configs/ros2_team.yaml --mode cooperative --metrics-dir outputs/<run>/nodes \\
        --ros-args -r __ns:=/go2_a -r __node:=coord_policy -p use_sim_time:=true
学習した協調の方策で動かす例:
    go2-ros2-team-coord --config configs/ros2_team.yaml --mode learned \\
        --policy-run-dir outputs/team_marl/mappo-shared-seed21 --policy-iteration 1000 --metrics-dir outputs/<run>/nodes \\
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
from go2_interfaces.msg import AgentStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from cooperative_go2.entities.command_expiry import ns_to_seconds, seconds_to_ns
from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    TeamMarlTrainingSettings,
    TeamRunSettings,
    load_ros2_team_settings,
    load_team_run_settings,
)
from cooperative_go2.interface_adapters.ros2_messages import StampedPose, make_twist_stamped, read_odometry, time_to_ns
from cooperative_go2.interface_adapters.ros2_metrics import LatencyStats, write_node_metrics
from cooperative_go2.interface_adapters.ros2_team_deciders import (
    LEARNED_MODE,
    LearnedTeamDecider,
    RuleBasedTeamDecider,
    TeamDecider,
    TeamDecision,
)
from cooperative_go2.interface_adapters.ros2_team_messages import (
    AgentStatusReport,
    make_agent_status,
    read_agent_status,
)
from cooperative_go2.interface_adapters.ros2_topics import (
    AGENT_STATUS_TOPIC,
    BASE_LINK_FRAME,
    CLEAR_CORRIDOR_ACTION,
    CMD_VEL_TOPIC,
    COMMAND_QOS,
    OBJECT_STATE_TOPIC,
    ODOM_TOPIC,
    SENSOR_QOS,
    TEAM_STATUS_QOS,
    WORLD_FRAME,
    robot_frame,
)
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.skrl_team_policy import load_team_policy, load_team_run_settings_copy
from cooperative_go2.use_cases.stamp_aligner import AlignedValues, StampAligner
from cooperative_go2.use_cases.team_policy_driver import TeamPolicyDriver
from cooperative_go2.use_cases.team_push_agent import CoordinationMode, RetreatArea

NODE_NAME = "coord_policy"
ROBOT_SOURCE = "odom"
BOX_SOURCE = "object_state"
SOURCES = (ROBOT_SOURCE, BOX_SOURCE)
MAX_PENDING_STAMPS = 10
# 依頼の取り消しや差し替えを確かめる実時間の間隔 [s]
GOAL_POLL_INTERVAL_S = 0.01
METRICS_FILENAME = "coord_policy_metrics.json"
# 受信待ちの区切り [s]（実時間）。SIGINT を受けてから終わるまでの遅れの上限になる
SPIN_TIMEOUT_S = 0.1
EXIT_INVALID_INPUT = 2
ROLL_COLUMN, PITCH_COLUMN = 0, 1
COOPERATIVE_TEAM_SIZE = 2
# 学習した方策の推論は 1 台分の小さな計算なので CPU で行う（シミュレーションノードと GPU を取り合わない）
CPU = "cpu"

logger = logging.getLogger(__name__)


class RobotNotInSceneError(ValueError):
    """ノードの Namespace のロボットが、シーンの設定の robots にいないことを表す。"""


class CooperativeTeamSizeError(ValueError):
    """協調のモードで、シーンのロボットが 2 台でないことを表す。相手の状態の判定は相手が 1 台だけの前提で作っている。"""


class PolicyConditionMismatchError(ValueError):
    """学習した方策の学習時の条件（シーン・判断の周期）が、ROS 2 の実行系の設定と違うことを表す。"""


@dataclass(frozen=True)
class TeamCoordSetup:
    """協調のノードの組み立てに使う値。

    Attributes:
        decider: 判断の中身。
        robot_names: シーンの robots の名前（同じ順）。
        period_s: 判断の周期 [s]（シミュレーション時間）。
        uses_partner: 相手の状態を購読して判断器に渡すか。
    """

    decider: TeamDecider
    robot_names: tuple[str, ...]
    period_s: float
    uses_partner: bool


@dataclass
class _ActiveGoal:
    handle: ServerGoalHandle
    start_stamp_ns: int
    finished: threading.Event
    # 次に判断する、依頼の時刻から数えた周期の番号。判断の時刻を開始時刻 + 番号 × 周期の格子にそろえる
    next_decision_index: int = 0
    verify_started_ns: int | None = None
    last_decision_ns: int | None = None
    final: TeamDecision | None = None


class TeamCoordNode:
    """協調のノード。受信とアクションの処理を複数のスレッドで並行して行い、判断の状態はロックで守る。"""

    def __init__(self, node: Node, setup: TeamCoordSetup) -> None:
        """ノードと組み立てに使う値を受け取り、購読・配信・アクションを作る。

        前提: ノードの Namespace のロボットが setup.robot_names にいること（_make_setup() で確かめる）。
        """
        self._node = node
        self._robot_name = node.get_namespace().strip("/")
        # 相手の状態として受け取るのは、シーンの設定の自分以外のロボットだけ（協調のモードでは 2 台なので 1 台）
        self._partner_names = frozenset(name for name in setup.robot_names if name != self._robot_name)
        self._decider = setup.decider
        self._period_ns = seconds_to_ns(setup.period_s)
        self._frame = robot_frame(node.get_namespace(), BASE_LINK_FRAME)
        self._lock = threading.Lock()
        self._aligner: StampAligner[StampedPose] = StampAligner(SOURCES, MAX_PENDING_STAMPS)
        self._latest: AlignedValues[StampedPose] | None = None
        self._goal: _ActiveGoal | None = None
        self._stopping = threading.Event()
        self.decision_time = LatencyStats()
        # 判断した時点での、最後に受け取った相手の状態の古さ（シミュレーション時間）。途絶えの判定時間を決める材料
        self.partner_status_age = LatencyStats()
        group = ReentrantCallbackGroup()
        self._cmd_vel = node.create_publisher(TwistStamped, CMD_VEL_TOPIC, COMMAND_QOS)
        self._status = node.create_publisher(AgentStatus, AGENT_STATUS_TOPIC, TEAM_STATUS_QOS)
        node.create_subscription(Odometry, ODOM_TOPIC, self._on_robot, SENSOR_QOS, callback_group=group)
        node.create_subscription(Odometry, OBJECT_STATE_TOPIC, self._on_box, SENSOR_QOS, callback_group=group)
        if setup.uses_partner:
            node.create_subscription(
                AgentStatus, AGENT_STATUS_TOPIC, self._on_agent_status, TEAM_STATUS_QOS, callback_group=group
            )
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
        self._put(ROBOT_SOURCE, robot)

    def _on_box(self, message: Odometry) -> None:
        box = read_odometry(message)
        self._put(BOX_SOURCE, box)

    def _on_agent_status(self, message: AgentStatus) -> None:
        report = read_agent_status(message)
        # 自分と、シーンにいないロボットの状態は使わない（別の構成のノードの配信を相手と取り違えないため）
        if report.agent_id not in self._partner_names:
            return
        with self._lock:
            self._decider.observe_partner(report)

    def _put(self, source: str, value: StampedPose) -> None:
        with self._lock:
            self._aligner.put(source, value.stamp_ns, value)
            aligned = self._aligner.pop_latest_complete_or_none()
            if aligned is None:
                return
            self._latest = aligned
            self._decide_if_due(aligned)

    def _execute(self, goal_handle: ServerGoalHandle) -> ClearCorridor.Result:
        request = goal_handle.request
        goal = _ActiveGoal(goal_handle, time_to_ns(request.header.stamp), threading.Event())
        retreat = RetreatArea(
            center_xy_m=(request.goal_position.x, request.goal_position.y), radius_m=request.goal_radius_m
        )
        with self._lock:
            self._decider.start(retreat, goal.start_stamp_ns)
            self._goal = goal
            # 試行の始めの状態が依頼より先に届いていれば、その状態で最初の判断をする
            if self._latest is not None:
                self._decide_if_due(self._latest)
        while not goal.finished.wait(GOAL_POLL_INTERVAL_S):
            if goal_handle.is_cancel_requested or self._goal is not goal:
                break
            if self._stopping.is_set():
                # ノードを終える（launch の終了）。依頼の結果は返せないので、待つのをやめて抜ける
                return ClearCorridor.Result(termination_reason="NODE_SHUTDOWN")
        with self._lock:
            # 次の依頼に差し替えられていたら、新しい依頼の指令を止めないよう停止の指令を出さない
            if self._goal is goal:
                self._goal = None
                self._publish_stop(goal)
        return self._finish(goal)

    def _finish(self, goal: _ActiveGoal) -> ClearCorridor.Result:
        result = ClearCorridor.Result()
        last = goal.start_stamp_ns if goal.last_decision_ns is None else goal.last_decision_ns
        result.elapsed_sim_s = ns_to_seconds(last - goal.start_stamp_ns)
        final = goal.final
        if final is not None and final.phase is TeamPhase.DONE:
            result.success = True
            result.termination_reason = "SUCCESS"
            goal.handle.succeed()
        elif final is not None and final.phase is TeamPhase.FAILED:
            result.success = False
            reason = final.failure_reason
            result.termination_reason = "FAILED" if reason is None else reason
            goal.handle.succeed()
        elif goal.handle.is_cancel_requested:
            result.termination_reason = "CANCELED"
            goal.handle.canceled()
        else:
            # 次の依頼に差し替えられた
            result.termination_reason = "PREEMPTED"
            goal.handle.abort()
        return result

    def _decide_if_due(self, aligned: AlignedValues[StampedPose]) -> None:
        # 前提: self._lock を持っている
        goal = self._goal
        if goal is None or goal.finished.is_set() or aligned.stamp_ns < goal.start_stamp_ns:
            return
        # 状態を取りこぼして格子の時刻より後の状態で判断しても、次の判断は格子の時刻に戻す（判断の時刻がずれ続けない）
        elapsed_ns = aligned.stamp_ns - goal.start_stamp_ns
        if elapsed_ns < goal.next_decision_index * self._period_ns:
            return
        goal.next_decision_index = elapsed_ns // self._period_ns + 1
        robot, box = (aligned.values[source].pose for source in SOURCES)
        started = time.perf_counter()
        decision = self._decider.decide(aligned.stamp_ns, robot, box)
        self.decision_time.add(time.perf_counter() - started)
        if decision.partner_age_s is not None:
            self.partner_status_age.add(decision.partner_age_s)
        goal.last_decision_ns = aligned.stamp_ns
        self._cmd_vel.publish(make_twist_stamped(aligned.stamp_ns, self._frame, decision.commands))
        self._status.publish(make_agent_status(self._report(aligned.stamp_ns, robot, decision), WORLD_FRAME))
        self._publish_feedback(goal, aligned.stamp_ns, decision, box.position_m[0, :2])
        if decision.phase in (TeamPhase.DONE, TeamPhase.FAILED):
            goal.final = decision
            goal.finished.set()

    def _publish_feedback(
        self, goal: _ActiveGoal, stamp_ns: int, decision: TeamDecision, box_xy_m: torch.Tensor
    ) -> None:
        if decision.phase is TeamPhase.VERIFY:
            if goal.verify_started_ns is None:
                goal.verify_started_ns = stamp_ns
        else:
            goal.verify_started_ns = None
        request = goal.handle.request
        retreat_xy = torch.tensor([request.goal_position.x, request.goal_position.y])
        feedback = ClearCorridor.Feedback()
        feedback.box_goal_distance_m = float(torch.linalg.vector_norm(box_xy_m - retreat_xy))
        verify_ns = 0 if goal.verify_started_ns is None else stamp_ns - goal.verify_started_ns
        feedback.in_goal_duration_s = ns_to_seconds(verify_ns)
        goal.handle.publish_feedback(feedback)

    def _report(self, stamp_ns: int, robot: RigidBodyPose, decision: TeamDecision) -> AgentStatusReport:
        planar = robot.to_planar()
        position = planar.position_xy_m[0]
        command = decision.commands[0]
        return AgentStatusReport(
            agent_id=self._robot_name,
            stamp_ns=stamp_ns,
            phase=decision.phase,
            push_position_xy_m=decision.push_position_xy_m,
            push_dir_xy=decision.push_dir_xy,
            position_xy_m=(float(position[0]), float(position[1])),
            yaw_rad=float(planar.yaw_rad[0]),
            command=(float(command[0]), float(command[1]), float(command[2])),
        )

    def _publish_stop(self, goal: _ActiveGoal) -> None:
        # 前提: self._lock を持っている。依頼が終わったら止まる。歩行方策のノードは停止の指令を受けて立ち止まる
        stamp_ns = goal.start_stamp_ns if goal.last_decision_ns is None else goal.last_decision_ns
        stop = torch.zeros((1, VELOCITY_COMMAND_SIZE))
        self._cmd_vel.publish(make_twist_stamped(stamp_ns, self._frame, stop))


@dataclass(frozen=True)
class _LoadedConfigs:
    run: TeamRunSettings
    policy: TeamMarlTrainingSettings | None


def _load_configs(args: argparse.Namespace) -> _LoadedConfigs:
    """ROS 2 の実行系の設定と、学習した方策で動かすならその学習時の設定を読み、条件がそろっていることを確かめる。

    Raises:
        FileNotFoundError, ConfigValidationError: 設定が無い・不正。
        PolicyConditionMismatchError: 学習時のシーンか判断の周期が ROS 2 の実行系の設定と違う、または学習時の通信の
            遅れが 1 周期でない。
    """
    run = load_team_run_settings(Path(load_ros2_team_settings(args.config).team_config))
    if args.mode != LEARNED_MODE:
        return _LoadedConfigs(run=run, policy=None)
    policy_run = load_team_run_settings_copy(args.policy_run_dir)
    policy = policy_run.settings
    # シミュレーションノードは ROS 2 の実行系の設定のシーンで物理を進めるので、学習時と同じシーン・周期でなければ比べられない。
    # シーンはファイルの場所ではなく中身（配置・寸法・質量・摩擦・接触の設定）を、学習の出力先に残した写しと比べる
    if policy_run.scene != run.scene:
        raise PolicyConditionMismatchError(
            f"学習時のシーン（{args.policy_run_dir} の写し）が実行系のシーン {run.settings.scene_config} と違う"
        )
    period_s = policy.task.episode.high_level_period_s
    if period_s != run.settings.team.high_level_period_s:
        raise PolicyConditionMismatchError(
            f"学習時の判断の周期 {period_s} s が実行系の周期 {run.settings.team.high_level_period_s} s と違う"
        )
    # 判断器は「判断の時刻より前の最新の相手の状態」を使うので、学習時の遅れが 1 周期のときにだけ学習時と同じ観測になる
    delay_steps = policy.task.communication_delay_steps()
    if delay_steps != 1:
        raise PolicyConditionMismatchError(f"学習時の通信の遅れが 1 周期でない（{delay_steps} 周期）")
    return _LoadedConfigs(run=run, policy=policy)


def _make_setup(args: argparse.Namespace, configs: _LoadedConfigs, robot_name: str) -> TeamCoordSetup:
    """ノードの Namespace のロボットの判断器を組み立てる。

    Raises:
        RobotNotInSceneError: ロボットがシーンの設定にいない。
        CooperativeTeamSizeError: 相手の状態を使うモードで、シーンのロボットが 2 台でない。
        CheckpointNotFoundError, CheckpointLoadError: 学習した方策を読み込めない。
    """
    names = tuple(robot.name for robot in configs.run.scene.scene.robots)
    if robot_name not in names:
        raise RobotNotInSceneError(f"Namespace のロボット {robot_name!r} がシーンの robots {names} にいない")
    uses_partner = args.mode != CoordinationMode.INDEPENDENT.value
    if uses_partner and len(names) != COOPERATIVE_TEAM_SIZE:
        raise CooperativeTeamSizeError(f"モード {args.mode} はロボット 2 台で動かす: {names}")
    index = names.index(robot_name)
    team = configs.run.settings.team
    policy = configs.policy
    if policy is None:
        decider: TeamDecider = RuleBasedTeamDecider(team, index, CoordinationMode(args.mode))
        return TeamCoordSetup(decider, names, team.high_level_period_s, uses_partner)
    actor = load_team_policy(args.policy_run_dir, args.policy_iteration, robot_name, CPU)
    driver = TeamPolicyDriver(actor.mean_commands, policy.task, robot_index=index)
    return TeamCoordSetup(LearnedTeamDecider(driver), names, policy.task.episode.high_level_period_s, uses_partner)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 の協調のノード（規則ベース、または学習した方策で箱を押す）")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="ROS 2 の実行系の設定（configs/ros2_team*.yaml 形式）。作業の設定はその team_config から読む",
    )  # fmt: skip
    parser.add_argument(
        "--mode", choices=[*(mode.value for mode in CoordinationMode), LEARNED_MODE], required=True,
        help="cooperative: 相手の状態を使い、相手が位置を直す間は押し続ける（E5）。cooperative_wait: 相手が押すのを"
        "やめたら止まって待つ。independent: 相手の状態を使わない（E3・E4）。learned: 学習した協調の方策で動かす",
    )  # fmt: skip
    parser.add_argument(
        "--policy-run-dir", type=Path, help="learned のとき必須。協調学習の出力先（outputs/team_marl/<run-name>）"
    )
    parser.add_argument("--policy-iteration", type=int, help="learned のとき必須。読み込むチェックポイントの更新の回数")
    parser.add_argument("--metrics-dir", type=Path, required=True, help="終了時に計測値を書くディレクトリ")
    args = parser.parse_args(argv)
    if args.mode == LEARNED_MODE and (args.policy_run_dir is None or args.policy_iteration is None):
        parser.error("--mode learned には --policy-run-dir と --policy-iteration が必要")
    return args


def main(argv: list[str] | None = None) -> None:
    """協調のノードを動かす。終了時に計測値を書く。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        configs = _load_configs(args)
    except (FileNotFoundError, ConfigValidationError, PolicyConditionMismatchError) as error:
        logger.error("協調のノードの設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    # 1 台分の小さな計算で、複数のスレッドを調整する手間の方が大きい
    torch.set_num_threads(1)
    # rclpy にシグナルを処理させない。rclpy が受信の途中でコンテキストを閉じると、受信の処理が内部エラーで落ちるため。
    # SIGINT は KeyboardInterrupt として受け、依頼を待つ処理と受信を止めてから閉じる
    rclpy.init(args=raw_argv, signal_handler_options=SignalHandlerOptions.NO)
    node = Node(NODE_NAME)
    try:
        coord_node = TeamCoordNode(node, _make_setup(args, configs, node.get_namespace().strip("/")))
    except (RobotNotInSceneError, CooperativeTeamSizeError, CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("%s", error)
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(EXIT_INVALID_INPUT)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        # 受信待ちを短く区切る。区切らないと、メッセージが途絶えたときに KeyboardInterrupt が上がらない
        while True:
            executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
    except KeyboardInterrupt:
        # launch の終了（SIGINT）で止める。正常に終えたときだけ計測値を書く（異常終了の計測値を残さない）
        logger.info("SIGINT を受けたので協調のノードを終える")
        robot = node.get_namespace().strip("/")
        write_node_metrics(
            args.metrics_dir / f"{robot}_{METRICS_FILENAME}",
            {
                "mode": args.mode,
                "decision_s": coord_node.decision_time.summary(),
                "partner_status_age_s": coord_node.partner_status_age.summary(),
            },
        )
    finally:
        coord_node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
