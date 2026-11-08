"""ROS 2 の協調のノードで、学習した協調の方策（IPPO・MAPPO の Actor）を動かす中身。

学習環境（Go2TeamMarlEnv）と同じ観測を組み立て、同じ前処理・後処理で速度指令を決め、同じ条件で作業の終わりを判定する。
学習時、相手の状態は 1 周期以上遅れて届き、同じ時刻に相手が決めた指令は見えない。ROS 2 では相手の同じ時刻の状態が
自分の判断より先に届くことがあるので、判断の時刻より前に配信された状態だけを使う。
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from cooperative_go2.entities.command_expiry import ns_to_seconds, seconds_to_ns
from cooperative_go2.entities.gait_termination import exceeds_tilt_limit
from cooperative_go2.entities.partner_channel import PartnerMessage, PartnerReception
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region
from cooperative_go2.entities.push_termination import (
    PushTerminationFlags,
    PushTerminationReason,
    classify_push_terminations,
    is_outside_arena,
)
from cooperative_go2.entities.team_marl_config import TeamMarlTaskConfig
from cooperative_go2.entities.team_marl_observation import TeamActorObservationInputs, build_team_actor_observation
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE

# 覚えておく相手の状態の数。判断の時刻より前の最新の状態を選べればよいので、数周期分あれば足りる
_PARTNER_HISTORY = 8


class NoActiveTaskError(RuntimeError):
    """作業を受け取る前に速度指令を求めたことを表す。"""


class TeamActorPolicy(Protocol):
    """Actor の観測 (n, TEAM_ACTOR_OBSERVATION_DIM) から速度指令 (n, 3) を出す推論専用の方策。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """速度指令（平均の行動）を返す。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。"""
        ...


class TeamTaskStatus(Enum):
    """1 回の判断のあとの作業の状況。"""

    RUNNING = "running"
    SUCCESS = "success"
    FALL = "fall"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIME_OUT = "time_out"


_STATUS_BY_REASON = {
    PushTerminationReason.NONE: TeamTaskStatus.RUNNING,
    PushTerminationReason.TIME_OUT: TeamTaskStatus.TIME_OUT,
    PushTerminationReason.SUCCESS: TeamTaskStatus.SUCCESS,
    PushTerminationReason.FALL: TeamTaskStatus.FALL,
    PushTerminationReason.OUT_OF_BOUNDS: TeamTaskStatus.OUT_OF_BOUNDS,
}


@dataclass(frozen=True)
class PartnerReport:
    """通信で受け取った相手の状態（/team/agent_status）。

    Attributes:
        stamp_ns: 相手がこの状態を決めたシミュレーション時刻 [ns]。
        position_xy_m: その時刻の相手の胴体の位置 (x, y) [m]。
        yaw_rad: その時刻の相手のヨー角 [rad]。
        command: その時刻に相手が決めた速度指令 (vx, vy, yaw_rate)。
    """

    stamp_ns: int
    position_xy_m: tuple[float, float]
    yaw_rad: float
    command: tuple[float, float, float]


@dataclass(frozen=True)
class TeamPolicyInputs:
    """1 台分の、速度指令を決める時点の状態。テンソルの先頭の次元は 1。

    Attributes:
        now_ns: 判断する状態のシミュレーション時刻 [ns]。作業を始めた時刻以降。
        robot: 自分の胴体の水平面の状態（ワールド座標系の速度）。
        robot_projected_gravity: 自分の本体座標系で表した重力方向 (1, 3)。
        robot_roll_pitch_deg: 自分の胴体のロールとピッチ (1, 2) [deg]。転倒の判定に使う。
        box: 箱の水平面の状態。
    """

    now_ns: int
    robot: PlanarBodyState
    robot_projected_gravity: torch.Tensor
    robot_roll_pitch_deg: torch.Tensor
    box: PlanarBodyState


@dataclass(frozen=True)
class TeamPolicyDecision:
    """1 回の判断の結果。

    Attributes:
        commands: 速度指令 (1, 3)。設定の範囲に切り詰めた値。
        status: 判断のあとの作業の状況。
        in_retreat_steps: 箱の中心が退避領域に入っていると続けて観測した回数（この判断を含む）。
        partner_age_s: 観測に使った相手の状態の古さ [s]。使える相手の状態が無ければ None。
    """

    commands: torch.Tensor
    status: TeamTaskStatus
    in_retreat_steps: int
    partner_age_s: float | None


@dataclass(frozen=True)
class _Task:
    retreat_xy_m: torch.Tensor
    retreat_radius_m: float
    start_ns: int


class TeamPolicyDriver:
    """1 台の Go2 の協調の Actor を、学習環境と同じ観測・指令の切り詰め・終わりの判定で動かす。

    - 観測は build_team_actor_observation で組み立てる。前回の指令には、範囲に切り詰めた後の指令を使う。
    - 相手の状態は、作業を始めた時刻以降で、判断の時刻より前に配信された最新のものを使う。古さは学習時と同じく
      max_age_s で頭打ちにし、使える状態が無ければ「届いていない」として観測に入れる。
    - 終わりの判定の条件は学習環境と同じ。箱の中心が退避領域に入ったと続けて required_hold_steps 回観測したら成功、
      自分が転倒の傾きを超えた・自分か箱が有効範囲の外に出たら失敗、作業の始めからエピソードの長さが経ったら時間切れ。
      2 台の判定を合わせるのはタスク管理のノードの役目なので、相手の転倒はここでは判定しない。
      転倒の傾きは判断の時刻の状態で見る。学習環境は物理ステップごとに見るので一時的な傾きも数えるが、ROS 2 の
      規則ベースの協調も判断の時刻で見るので、ROS 2 の比較の条件はそろう。
    前提: decide() を高レベル方策の周期（task.episode.high_level_period_s）の格子の時刻ごとに呼ぶ。
    """

    def __init__(self, policy: TeamActorPolicy, task: TeamMarlTaskConfig, robot_index: int) -> None:
        """学習済みの Actor、その Actor を学習したときの環境の設定、シーンの robots の中での自分の番号を受け取る。

        自分の側（観測に入れる左右）は、学習時と同じく task.agent_side_signs[robot_index] から決める。

        Raises:
            IndexError: robot_index に対応する側が task.agent_side_signs に無い。
        """
        if not 0 <= robot_index < task.num_agents:
            raise IndexError(f"robot_index {robot_index} はロボット {task.num_agents} 台の番号の外")
        self._policy = policy
        self._task = task
        self._side_sign = task.agent_side_signs[robot_index]
        limits = task.commands.as_ranges()
        self._command_low = torch.tensor([[limit.low for limit in limits]])
        self._command_high = torch.tensor([[limit.high for limit in limits]])
        self._active: _Task | None = None
        self._partner_reports: deque[PartnerReport] = deque(maxlen=_PARTNER_HISTORY)
        self._last_commands = torch.zeros((1, VELOCITY_COMMAND_SIZE))
        self._hold_steps = torch.zeros(1, dtype=torch.int64)
        self._period_ns = seconds_to_ns(task.episode.high_level_period_s)

    def start(self, retreat_xy_m: tuple[float, float], retreat_radius_m: float, start_ns: int) -> None:
        """新しい作業を始める。前回の指令と、退避領域に入っていた回数を 0 に戻す。

        受け取った相手の状態は捨てない。タスク管理は 2 台に順に依頼を送るので、先に始めた相手の最初の状態が、
        自分の作業を始める前に届くことがあるため。前の作業の状態は、時刻が start_ns より前なので使われない。

        Args:
            retreat_xy_m: 退避領域の中心 (x, y) [m]（ワールド座標系）。
            retreat_radius_m: 退避領域の半径 [m]。
            start_ns: 作業を始めた（依頼の）シミュレーション時刻 [ns]。これより前に配信された相手の状態は使わない。
        """
        self._active = _Task(torch.tensor([retreat_xy_m]), retreat_radius_m, start_ns)
        self._last_commands = torch.zeros((1, VELOCITY_COMMAND_SIZE))
        self._hold_steps = torch.zeros(1, dtype=torch.int64)

    def observe_partner(self, report: PartnerReport) -> None:
        """通信で受け取った相手の状態を覚える。どれを判断に使うかは decide() のときに、作業の始めの時刻以降で、
        判断の時刻より前に配信されたものから選ぶ。"""
        self._partner_reports.append(report)

    def decide(self, inputs: TeamPolicyInputs) -> TeamPolicyDecision:
        """速度指令を決め、作業の終わりの判定を 1 回進める。

        Raises:
            NoActiveTaskError: start() で作業を受け取っていない。
        """
        active = self._active
        if active is None:
            raise NoActiveTaskError("作業を受け取る前に速度指令を求めた")
        reception, partner_age_s = self._partner_reception(inputs.now_ns, active.start_ns)
        observation = build_team_actor_observation(
            TeamActorObservationInputs(
                robot=inputs.robot,
                robot_projected_gravity=inputs.robot_projected_gravity,
                box=inputs.box,
                goal_xy_m=active.retreat_xy_m,
                last_commands=self._last_commands,
                side_sign=self._side_sign,
                partner=reception,
            ),
            self._task.observation,
            self._task.communication.max_age_s,
        )
        with torch.no_grad():
            raw_commands = self._policy(observation)
        commands = torch.clamp(raw_commands, self._command_low, self._command_high)
        self._last_commands = commands
        status = self._judge(inputs, active)
        return TeamPolicyDecision(
            commands=commands, status=status, in_retreat_steps=int(self._hold_steps[0]), partner_age_s=partner_age_s
        )

    def _partner_reception(self, now_ns: int, start_ns: int) -> tuple[PartnerReception, float | None]:
        usable = [report for report in self._partner_reports if start_ns <= report.stamp_ns < now_ns]
        now_s = torch.tensor([ns_to_seconds(now_ns - start_ns)])
        if not usable:
            message = PartnerMessage(
                position_xy_m=torch.zeros((1, 2)), yaw_rad=torch.zeros(1),
                command=torch.zeros((1, VELOCITY_COMMAND_SIZE)), stamp_s=torch.zeros(1),
            )  # fmt: skip
            max_age_s = self._task.communication.max_age_s
            return PartnerReception(message=message, received=torch.zeros(1, dtype=torch.bool),
                                    age_s=torch.full((1,), max_age_s)), None  # fmt: skip
        latest = max(usable, key=lambda report: report.stamp_ns)
        # 学習環境と同じく、エピソード（作業）の始めから数えた時刻で古さを求める
        stamp_s = torch.tensor([ns_to_seconds(latest.stamp_ns - start_ns)])
        message = PartnerMessage(
            position_xy_m=torch.tensor([latest.position_xy_m]),
            yaw_rad=torch.tensor([latest.yaw_rad]),
            command=torch.tensor([latest.command]),
            stamp_s=stamp_s,
        )
        age_s = now_s - stamp_s
        capped_age_s = torch.clamp(age_s, max=self._task.communication.max_age_s)
        reception = PartnerReception(message=message, received=torch.ones(1, dtype=torch.bool), age_s=capped_age_s)
        return reception, float(age_s[0])

    def _judge(self, inputs: TeamPolicyInputs, active: _Task) -> TeamTaskStatus:
        task = self._task
        box_xy = inputs.box.position_xy_m
        in_retreat = is_in_goal_region(box_xy, active.retreat_xy_m, active.retreat_radius_m)
        self._hold_steps = advance_goal_hold_steps(self._hold_steps, in_retreat)
        episode = task.episode
        roll_pitch_yaw = torch.cat((inputs.robot_roll_pitch_deg, torch.zeros((1, 1))), dim=1)
        flags = PushTerminationFlags(
            # 作業の始めから進んだ周期の数が、学習環境のエピソードのステップ数にあたる。状態を取りこぼして判断を
            # 飛ばしても、時間切れの時刻はずれない
            time_out=torch.tensor(
                [(inputs.now_ns - active.start_ns) // self._period_ns >= episode.max_episode_length_steps()]
            ),
            succeeded=has_held_goal(self._hold_steps, task.required_hold_steps()),
            fallen=exceeds_tilt_limit(roll_pitch_yaw, episode.fall_roll_deg, episode.fall_pitch_deg),
            out_of_bounds=is_outside_arena(inputs.robot.position_xy_m, task.arena)
            | is_outside_arena(box_xy, task.arena),
            diverged=torch.zeros(1, dtype=torch.bool),
        )
        return _STATUS_BY_REASON[PushTerminationReason(int(classify_push_terminations(flags)[0]))]
