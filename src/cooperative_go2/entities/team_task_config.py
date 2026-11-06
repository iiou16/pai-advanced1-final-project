"""規則ベースで箱を押す作業（1 台・2 台）の設定と、試行の始めのロボットの開始位置の引き方。

既定値はここに持たず、設定ファイル（configs/team_rule_based.yaml など）にすべての値を書く。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError, is_integer_ratio
from cooperative_go2.entities.coordination_state_machine import CoordinationConfig
from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.entities.team_push_control import TeamPushConfig
from cooperative_go2.entities.team_readiness import TeamReadinessConfig



class InvalidTeamTaskConfigError(InvalidConfigError):
    """規則ベースで箱を押す作業の設定値が不変条件を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidTeamTaskConfigError)


@dataclass(frozen=True)
class TeamResetConfig:
    """試行の始めのロボットの開始位置の散らし方。向きはシーンの設定どおり（+x）にする。

    各ロボットは、シーンの設定の初期位置から、押す方向の逆（-x）へ start_back_m、横（y）へ ±start_lateral_jitter_m の
    範囲で一様にずらした位置から始める。ロボットごとに独立に引くので、押す位置に着く時刻がずれる。

    Attributes:
        start_back_m: 後ろへずらす距離の範囲 [m]。下限は 0 以上。
        start_lateral_jitter_m: 横へずらす距離の上限 [m]。0 以上。2 台がぶつからない幅にする。
    """

    start_back_m: ValueRange
    start_lateral_jitter_m: float

    def __post_init__(self) -> None:
        _check.non_negative("reset.start_back_m.low", self.start_back_m.low)
        _check.non_negative("reset.start_lateral_jitter_m", self.start_lateral_jitter_m)


@dataclass(frozen=True)
class TaskManagerConfig:
    """タスク管理（ClearCorridor の依頼を各ロボットへ割り振るノード）の設定。

    Attributes:
        timeout_s: 依頼の時刻から、各ロボットの結果がそろわなければ時間切れにするまでの時間 [s]（シミュレーション時間）。
            ロボットの作業全体の時間切れ（coordination.timeouts.task_s）より長くする。ロボットが自分で失敗を
            返せなくなった（ノードが止まった、など）ときの最後の歯止めにするため。
    """

    timeout_s: float

    def __post_init__(self) -> None:
        _check.positive("task_manager.timeout_s", self.timeout_s)


@dataclass(frozen=True)
class TeamRuleBasedConfig:
    """規則ベースで箱を押す作業の設定一式。

    Attributes:
        high_level_period_s: 状態機械を進め、速度指令を決める周期 [s]（シミュレーション時間）。正の値。
        fall_tilt_deg: ロールかピッチの絶対値がこの角度 [deg] を超えたら転倒とみなす。正の値。歩行方策の学習時の
            終了条件より緩くしない（それ以上傾いた状態を歩行方策は知らない）。
        push_lateral_offsets_m: ロボットごとの押す位置の、押す方向の線からの横のずれ [m]。シーンの設定の robots と
            同じ順・同じ数。押す方向の左が正。押す側を前もって決めておき、通信では割り当てない。
        push_control: 押す位置・判定・速度指令のパラメータ。
        coordination: 状態機械の時間切れと VERIFY の時間。
        team_readiness: 2 台の協調で、相手の状態から準備を判定する時間。
        reset: 試行の始めの開始位置の散らし方。
        task_manager: タスク管理の設定。
    """

    high_level_period_s: float
    fall_tilt_deg: float
    push_lateral_offsets_m: tuple[float, ...]
    push_control: TeamPushConfig
    coordination: CoordinationConfig
    team_readiness: TeamReadinessConfig
    reset: TeamResetConfig
    task_manager: TaskManagerConfig

    def __post_init__(self) -> None:
        _check.positive("high_level_period_s", self.high_level_period_s)
        _check.positive("fall_tilt_deg", self.fall_tilt_deg)
        if not self.push_lateral_offsets_m:
            raise InvalidTeamTaskConfigError("push_lateral_offsets_m には 1 台以上の値が必要")
        for index, offset in enumerate(self.push_lateral_offsets_m):
            _check.finite(f"push_lateral_offsets_m[{index}]", offset)
        if self.task_manager.timeout_s <= self.coordination.timeouts.task_s:
            raise InvalidTeamTaskConfigError(
                f"task_manager.timeout_s ({self.task_manager.timeout_s}) は coordination.timeouts.task_s "
                f"({self.coordination.timeouts.task_s}) より長くする"
            )

    def gait_steps_per_high_level_step(self, gait_control_dt_s: float) -> int:
        """状態機械の 1 周期の間に、歩行方策を何回呼ぶか（物理を何制御周期進めるか）を返す。

        Raises:
            InvalidTeamTaskConfigError: 周期が歩行方策の制御周期の整数倍でない。
        """
        if not is_integer_ratio(self.high_level_period_s, gait_control_dt_s):
            raise InvalidTeamTaskConfigError(
                f"high_level_period_s ({self.high_level_period_s}) は歩行方策の制御周期 ({gait_control_dt_s}) の整数倍にする"
            )
        return round(self.high_level_period_s / gait_control_dt_s)


def sample_team_start_positions(
    initial_xy_m: tuple[tuple[float, ...], ...], reset: TeamResetConfig, num_envs: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """ロボットごとの開始位置 (num_envs, 2) [m] を引く。ロボットの順に、x のずれ・y のずれの順で引く。

    Args:
        initial_xy_m: シーンの設定の、ロボットごとの初期位置 (x, y) [m]。
        reset: 散らし方。
        num_envs: 並列環境（試行）の数。
        generator: 乱数生成器。テンソルはこの生成器のデバイスに置く。
    """
    positions = []
    back = reset.start_back_m
    for x_m, y_m in initial_xy_m:
        back_m = back.low + (back.high - back.low) * torch.rand(num_envs, generator=generator, device=generator.device)
        unit = torch.rand(num_envs, generator=generator, device=generator.device)
        lateral_m = (2.0 * unit - 1.0) * reset.start_lateral_jitter_m
        positions.append(torch.stack((x_m - back_m, y_m + lateral_m), dim=1))
    return tuple(positions)
