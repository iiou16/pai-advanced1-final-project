"""ROS 2 の協調のノードが使う、1 台のロボットの判断器（規則ベースの協調と、学習した協調の方策）。

協調のノードは、受信の時刻合わせ・作業の依頼（ClearCorridor）・判断の周期・配信を受け持ち、判断の中身をこの判断器に任せる。
どちらの判断器も、同じ型の入力（自分と箱の状態、通信で受け取った相手の状態）から、同じ型の判断（速度指令と段階）を返す。
"""

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.command_expiry import ns_to_seconds
from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.push_trial import PlanarPose
from cooperative_go2.entities.quaternion import roll_pitch_deg_from_quat_wxyz
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.entities.team_task_config import TeamRuleBasedConfig
from cooperative_go2.interface_adapters.ros2_team_messages import AgentStatusReport
from cooperative_go2.use_cases.team_policy_driver import (
    PartnerReport,
    TeamPolicyDriver,
    TeamPolicyInputs,
    TeamTaskStatus,
)
from cooperative_go2.use_cases.team_push_agent import (
    CoordinationMode,
    RetreatArea,
    TeamPushAgent,
    TeamPushAgentInputs,
)

# 学習した協調の方策で動かすモードの名前（規則ベースの CoordinationMode の値と並べて選ぶ）
LEARNED_MODE = "learned"


@dataclass(frozen=True)
class TeamDecision:
    """1 回の判断の結果。

    Attributes:
        commands: 速度指令 (1, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]（本体座標系）。
        phase: 判断のあとの段階。DONE か FAILED なら作業を終える。VERIFY の間は、箱が退避領域に入ったまま続いている。
        failure_reason: FAILED の理由の名前。FAILED でなければ None。
        push_position_xy_m: 相手に知らせる押す位置 (x, y) [m]。
        push_dir_xy: 相手に知らせる押す方向の単位ベクトル (x, y)。
        partner_age_s: 判断に使った相手の状態の古さ [s]（シミュレーション時間）。相手の状態を使わなかったら None。
    """

    commands: torch.Tensor
    phase: TeamPhase
    failure_reason: str | None
    push_position_xy_m: tuple[float, float]
    push_dir_xy: tuple[float, float]
    partner_age_s: float | None


class TeamDecider(Protocol):
    """協調のノードの判断の中身。前提: 作業ごとに start() を呼んでから、判断の周期の格子の時刻ごとに decide() を呼ぶ。"""

    def start(self, retreat: RetreatArea, start_ns: int) -> None:
        """新しい作業（箱を retreat へ移す）を、シミュレーション時刻 start_ns [ns] に始める。"""
        ...

    def observe_partner(self, report: AgentStatusReport) -> None:
        """通信で受け取った相手の状態を覚える。"""
        ...

    def decide(self, stamp_ns: int, robot: RigidBodyPose, box: RigidBodyPose) -> TeamDecision:
        """時刻 stamp_ns [ns] の自分と箱の状態から判断する。"""
        ...


def _planar_pose(body: PlanarBodyState) -> PlanarPose:
    position = body.position_xy_m[0]
    return PlanarPose(x_m=float(position[0]), y_m=float(position[1]), yaw_rad=float(body.yaw_rad[0]))


class RuleBasedTeamDecider:
    """規則ベースの協調の判断（TeamPushAgent）。並列環境の確認と同じ判断を、ROS 2 で受け取った状態で動かす。"""

    def __init__(self, team: TeamRuleBasedConfig, robot_index: int, mode: CoordinationMode) -> None:
        """作業の設定・シーンの robots の中での自分の番号・協調のモードを受け取る。

        Raises:
            IndexError: robot_index に対応する押す側の割り当てが設定に無い。
        """
        self._team = team
        self._mode = mode
        self._agent = TeamPushAgent(team, robot_index, mode)
        self._partner: AgentStatusReport | None = None

    def start(self, retreat: RetreatArea, start_ns: int) -> None:
        """状態機械を INIT に戻して新しい作業を始める。"""
        self._agent.start(retreat, now_s=ns_to_seconds(start_ns))

    def observe_partner(self, report: AgentStatusReport) -> None:
        """相手の状態を覚える。遅れて届いた古い状態で、新しい状態を上書きしない。"""
        if self._partner is None or report.stamp_ns >= self._partner.stamp_ns:
            self._partner = report

    def decide(self, stamp_ns: int, robot: RigidBodyPose, box: RigidBodyPose) -> TeamDecision:
        """状態機械を 1 回進め、その段階の速度指令を返す。通信なしの独立制御では相手の状態を使わない。"""
        now_s = ns_to_seconds(stamp_ns)
        partner = None
        partner_age_s = None
        if self._mode is not CoordinationMode.INDEPENDENT and self._partner is not None:
            partner = self._partner.to_partner_status()
            partner_age_s = now_s - partner.stamp_s
        roll_pitch = roll_pitch_deg_from_quat_wxyz(robot.quat_wxyz)[0]
        fallen = bool(roll_pitch.abs().max() > self._team.fall_tilt_deg)
        decision = self._agent.decide(
            TeamPushAgentInputs(
                now_s=now_s,
                robot=_planar_pose(robot.to_planar()),
                box=_planar_pose(box.to_planar()),
                partner=partner,
                task_assigned=True,
                fallen=fallen,
            )
        )
        reason = decision.state.failure_reason
        return TeamDecision(
            commands=decision.commands,
            phase=decision.state.phase,
            failure_reason=None if reason is None else reason.name,
            push_position_xy_m=decision.push_position_xy_m,
            push_dir_xy=decision.push_dir_xy,
            partner_age_s=partner_age_s,
        )


class LearnedTeamDecider:
    """学習した協調の方策（TeamPolicyDriver）。学習環境と同じ観測を、ROS 2 で受け取った状態から作る。

    学習した方策には段階が無いので、作業の間は APPROACH を、箱が退避領域に入ったまま続いている間は VERIFY を配信し、
    成功なら DONE、転倒・離脱・時間切れなら FAILED にする。押し始めの時刻（PUSH に入った時刻）は記録されない。
    押す位置と方向の欄には、学習した方策が押す位置を持たないので、自分の現在の位置と向きを入れる。
    """

    def __init__(self, driver: TeamPolicyDriver) -> None:
        """学習した方策を動かす中身を受け取る。"""
        self._driver = driver

    def start(self, retreat: RetreatArea, start_ns: int) -> None:
        """前回の指令と受け取った相手の状態を捨てて、新しい作業を始める。"""
        self._driver.start(retreat.center_xy_m, retreat.radius_m, start_ns)

    def observe_partner(self, report: AgentStatusReport) -> None:
        """相手の状態（位置・向き・指令）を覚える。"""
        self._driver.observe_partner(
            PartnerReport(
                stamp_ns=report.stamp_ns,
                position_xy_m=report.position_xy_m,
                yaw_rad=report.yaw_rad,
                command=report.command,
            )
        )

    def decide(self, stamp_ns: int, robot: RigidBodyPose, box: RigidBodyPose) -> TeamDecision:
        """学習した方策で速度指令を決め、作業の終わりを判定する。"""
        robot_planar = robot.to_planar()
        decision = self._driver.decide(
            TeamPolicyInputs(
                now_ns=stamp_ns,
                robot=robot_planar,
                robot_projected_gravity=robot.projected_gravity(),
                robot_roll_pitch_deg=roll_pitch_deg_from_quat_wxyz(robot.quat_wxyz),
                box=box.to_planar(),
            )
        )
        phase, reason = _phase_of(decision.status, decision.in_retreat_steps)
        pose = _planar_pose(robot_planar)
        return TeamDecision(
            commands=decision.commands,
            phase=phase,
            failure_reason=reason,
            push_position_xy_m=(pose.x_m, pose.y_m),
            push_dir_xy=(math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)),
            partner_age_s=decision.partner_age_s,
        )


def _phase_of(status: TeamTaskStatus, in_retreat_steps: int) -> tuple[TeamPhase, str | None]:
    if status is TeamTaskStatus.RUNNING:
        return (TeamPhase.VERIFY if in_retreat_steps > 0 else TeamPhase.APPROACH), None
    if status is TeamTaskStatus.SUCCESS:
        return TeamPhase.DONE, None
    return TeamPhase.FAILED, status.name
