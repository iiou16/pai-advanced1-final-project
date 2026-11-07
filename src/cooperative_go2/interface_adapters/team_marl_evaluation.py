"""学習した協調の方策を、ROS 2 を通さずに並列環境で 1 試行ずつ評価する（E6・E7）。

規則ベースの協調の並列環境での確認（scripts/p4_coordination_check/run_team_push.py）と同じ列で試行ごとの結果を返し、
同じシード・同じ並列環境の数なら同じ開始位置と関節角から始める。Actor は平均の行動（確率的に引かない）で動かす。

押し始めの時刻は、規則ベースでは状態機械が PUSH に入った時刻だが、学習した方策には段階が無いので、
ここでは各ロボットが初めて箱に触れた時刻（シミュレーションの接触判定）とする。定義が違うので列の名前を変える。
"""

import math
from dataclasses import dataclass

import torch

from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.interface_adapters.skrl_team import TeamPolicy
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv, TeamStepOutcome


@dataclass(frozen=True)
class TeamMarlTrialRow:
    """1 試行（並列環境 1 つの最初のエピソード）の結果。CSV の 1 行になる。

    Attributes:
        env: 並列環境の番号。
        success: 箱の中心が退避領域に入ったまま所定の時間続いた。
        termination_reason: 終了理由（PushTerminationReason の名前）。
        corridor_opened: 試行中に通路が開いた。
        open_time_s: 通路が初めて開いた時刻 [s]。開かなければ None。
        end_time_s: 試行を終えた時刻 [s]。
        first_box_contact_times_s: ロボットごとの、初めて箱に触れた時刻 [s]（| でつなぐ。触れなければ -）。
        first_box_contact_spread_s: 初めて箱に触れた時刻の最大と最小の差 [s]。全員が触れなければ None。
        box_dx_m: 箱の +x 方向の移動量 [m]。
        box_max_abs_yaw_deg: 試行中の、押す方向（箱から退避領域の中心への向き）から見た箱の回転の絶対値の最大 [deg]。
        falls: 転倒したロボットの数。
    """

    env: int
    success: bool
    termination_reason: str
    corridor_opened: bool
    open_time_s: float | None
    end_time_s: float
    first_box_contact_times_s: str
    first_box_contact_spread_s: float | None
    box_dx_m: float
    box_max_abs_yaw_deg: float
    falls: int


def _box_yaw_error(box_xy_m: torch.Tensor, box_yaw_rad: torch.Tensor, retreat_xy_m: torch.Tensor) -> torch.Tensor:
    # 箱は 180° 回しても同じ形なので、押す方向とのなす角を [-π/2, π/2) に畳む（規則ベースの確認と同じ定義）
    to_retreat = retreat_xy_m - box_xy_m
    push_angle = torch.atan2(to_retreat[:, 1], to_retreat[:, 0])
    return torch.remainder(box_yaw_rad - push_angle + math.pi / 2, math.pi) - math.pi / 2


class _TrialProgress:
    """環境ごとの、最初のエピソードの途中経過。"""

    def __init__(self, num_envs: int, num_robots: int, box_start_x_m: torch.Tensor) -> None:
        self.finished = [False] * num_envs
        self.reasons = [PushTerminationReason.NONE] * num_envs
        self.end_time_s = [0.0] * num_envs
        self.open_time_s: list[float | None] = [None] * num_envs
        self.first_contact_s: list[list[float | None]] = [[None] * num_robots for _ in range(num_envs)]
        self.box_start_x_m = box_start_x_m.cpu()
        self.box_end_x_m = self.box_start_x_m.clone()
        self.max_abs_yaw_rad = torch.zeros(num_envs)
        self.fallen = torch.zeros(num_robots, num_envs, dtype=torch.bool)


def evaluate_team_policies(env: Go2TeamMarlEnv, policies: tuple[TeamPolicy, ...]) -> list[TeamMarlTrialRow]:
    """全環境で最初のエピソードを 1 回ずつ動かし、環境ごとの結果を返す。

    前提: env は作った直後（最初のリセットのまま）であること。終わった環境は次のエピソードに進むが、記録しない。

    Args:
        env: 評価する並列環境。
        policies: ロボットごとの Actor（シーンの robots と同じ順）。
    """
    num_envs = env.num_envs
    period_s = env.cfg.episode.high_level_period_s
    retreat_xy = env.goal_xy_m
    progress = _TrialProgress(num_envs, env.num_agents, env.box_planar.position_xy_m[:, 0].clone())
    observations = env.observations
    for step in range(1, env.max_episode_length + 1):
        with torch.no_grad():
            actions = tuple(policy.mean_commands(obs) for policy, obs in zip(policies, observations, strict=True))
        outcome = env.step(actions)
        observations = outcome.observations
        _record_step(progress, outcome, retreat_xy, round(step * period_s, 6))
        if all(progress.finished):
            break
    return [_row(progress, env_index) for env_index in range(num_envs)]


def _record_step(progress: _TrialProgress, outcome: TeamStepOutcome, retreat_xy: torch.Tensor, time_s: float) -> None:
    snapshot = outcome.pre_reset
    footprint = snapshot.box_footprint
    yaw_error = _box_yaw_error(footprint.center_xy_m, footprint.yaw_rad, retreat_xy).abs().cpu()
    box_x = footprint.center_xy_m[:, 0].cpu()
    opened = snapshot.corridor_open.cpu()
    contacts = [contact.cpu() for contact in snapshot.box_contacts]
    fallen = torch.stack([robot_fallen.cpu() for robot_fallen in snapshot.fallen])
    done = outcome.done.cpu()
    reasons = outcome.termination_reasons.cpu()
    for env_index in range(len(progress.finished)):
        if progress.finished[env_index]:
            continue
        progress.max_abs_yaw_rad[env_index] = max(
            float(progress.max_abs_yaw_rad[env_index]), float(yaw_error[env_index])
        )
        progress.box_end_x_m[env_index] = box_x[env_index]
        progress.fallen[:, env_index] |= fallen[:, env_index]
        if progress.open_time_s[env_index] is None and bool(opened[env_index]):
            progress.open_time_s[env_index] = time_s
        for robot_index, contact in enumerate(contacts):
            if progress.first_contact_s[env_index][robot_index] is None and bool(contact[env_index]):
                progress.first_contact_s[env_index][robot_index] = time_s
        if bool(done[env_index]):
            progress.finished[env_index] = True
            progress.reasons[env_index] = PushTerminationReason(int(reasons[env_index]))
            progress.end_time_s[env_index] = time_s


def _row(progress: _TrialProgress, env_index: int) -> TeamMarlTrialRow:
    contacts = progress.first_contact_s[env_index]
    touched = [time_s for time_s in contacts if time_s is not None]
    reason = progress.reasons[env_index]
    return TeamMarlTrialRow(
        env=env_index,
        success=reason is PushTerminationReason.SUCCESS,
        termination_reason=reason.name,
        corridor_opened=progress.open_time_s[env_index] is not None,
        open_time_s=progress.open_time_s[env_index],
        end_time_s=progress.end_time_s[env_index],
        first_box_contact_times_s="|".join("-" if time_s is None else f"{time_s:.1f}" for time_s in contacts),
        first_box_contact_spread_s=round(max(touched) - min(touched), 2) if len(touched) == len(contacts) else None,
        box_dx_m=round(float(progress.box_end_x_m[env_index] - progress.box_start_x_m[env_index]), 3),
        box_max_abs_yaw_deg=round(math.degrees(float(progress.max_abs_yaw_rad[env_index])), 1),
        falls=int(progress.fallen[:, env_index].sum()),
    )
