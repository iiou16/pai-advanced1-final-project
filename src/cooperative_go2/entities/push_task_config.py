"""1 台の箱押しの学習環境（エピソードの進め方・高レベル方策の指令・リセット時の配置）の設定の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/push_single.yaml）にすべての値を書く。
各クラスは生成時に値の不変条件を検証し、違反すれば InvalidPushConfigError を送出する。
歩行方策の学習時の設定と突き合わせる検証は require_compatible_with_gait() で行う。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker, ceil_ratio, is_integer_ratio
from cooperative_go2.entities.gait_config import GaitConfig, ValueRange, VelocityCommandConfig
from cooperative_go2.entities.push_config import (
    InvalidPushConfigError,
    PushArenaConfig,
    PushDivergenceConfig,
    PushObservationScales,
    PushRewardConfig,
    PushSuccessConfig,
    RuleBasedPushConfig,
)

_check = ConfigValueChecker(InvalidPushConfigError)


@dataclass(frozen=True)
class PushEpisodeConfig:
    """エピソードの進め方。

    Attributes:
        num_envs: 並列環境の数。正の整数。
        high_level_period_s: 高レベル方策が速度指令を出す周期 [s]。歩行方策の制御周期の整数倍にする。
        episode_length_s: 1 エピソードの長さ [s]。正の値。
        fall_roll_deg: ロボットのロールの絶対値がこれを超えたら転倒とする [deg]。正の値。
        fall_pitch_deg: ロボットのピッチの絶対値がこれを超えたら転倒とする [deg]。正の値。
    """

    num_envs: int
    high_level_period_s: float
    episode_length_s: float
    fall_roll_deg: float
    fall_pitch_deg: float

    def __post_init__(self) -> None:
        _check.positive("episode.num_envs", self.num_envs)
        _check.positive("episode.high_level_period_s", self.high_level_period_s)
        _check.positive("episode.episode_length_s", self.episode_length_s)
        _check.positive("episode.fall_roll_deg", self.fall_roll_deg)
        _check.positive("episode.fall_pitch_deg", self.fall_pitch_deg)

    def max_episode_length_steps(self) -> int:
        """1 エピソードの長さを高レベル方策のステップ数で返す（端数は切り上げ）。"""
        return ceil_ratio(self.episode_length_s, self.high_level_period_s)


@dataclass(frozen=True)
class PushCommandLimits:
    """高レベル方策が出せる速度指令の範囲。方策の出力はこの範囲に切り詰めてから歩行方策に渡す。

    Attributes:
        vx_mps: 前後方向の速度 [m/s]。
        vy_mps: 左右方向の速度 [m/s]。
        yaw_rate_radps: ヨー角速度 [rad/s]。
    """

    vx_mps: ValueRange
    vy_mps: ValueRange
    yaw_rate_radps: ValueRange

    def as_ranges(self) -> tuple[ValueRange, ValueRange, ValueRange]:
        """(vx, vy, yaw_rate) の順の範囲を返す。速度指令のテンソルの列の順序と同じ。"""
        return (self.vx_mps, self.vy_mps, self.yaw_rate_radps)


@dataclass(frozen=True)
class PushResetConfig:
    """エピソードの始めの箱と目標の配置。ロボットは歩行方策の学習時の初期位置・姿勢に置く。

    Attributes:
        box_xy_jitter_m: シーンの設定の箱の初期位置からずらす量の上限 [m]。x・y それぞれ [-値, 値] から引く。0 以上。
        goal_distance_m: 箱の初期位置から目標の中心までの距離 [m] の範囲。下限は正の値。
        goal_direction_deg: 箱の初期位置から見た目標の方向 [deg] の範囲。ワールドの +x から反時計回り。
    """

    box_xy_jitter_m: float
    goal_distance_m: ValueRange
    goal_direction_deg: ValueRange

    def __post_init__(self) -> None:
        _check.non_negative("reset.box_xy_jitter_m", self.box_xy_jitter_m)
        _check.positive("reset.goal_distance_m.low", self.goal_distance_m.low)


@dataclass(frozen=True)
class PushTaskConfig:
    """1 台の箱押しの学習環境に必要な、学習アルゴリズムによらない設定一式。"""

    episode: PushEpisodeConfig
    commands: PushCommandLimits
    reset: PushResetConfig
    success: PushSuccessConfig
    arena: PushArenaConfig
    divergence: PushDivergenceConfig
    observation: PushObservationScales
    reward: PushRewardConfig

    def gait_steps_per_high_level_step(self, gait_control_dt_s: float) -> int:
        """高レベル方策の 1 ステップの間に、歩行方策を何回呼ぶか（物理を何制御周期進めるか）を返す。

        前提: require_compatible_with_gait() で整数倍であることを確かめていること。
        """
        return round(self.episode.high_level_period_s / gait_control_dt_s)

    def required_hold_steps(self) -> int:
        """成功判定に必要な、目標領域に入った状態を続ける高レベル方策のステップ数を返す。"""
        return self.success.required_hold_steps(self.episode.high_level_period_s)


def require_compatible_with_gait(task: PushTaskConfig, gait: GaitConfig) -> None:
    """箱押しの設定が、固定して使う歩行方策の学習時の設定と矛盾しないことを確かめる。

    - 高レベル方策の周期が、歩行方策の制御周期の整数倍であること
    - 高レベル方策が出せる速度指令の範囲が、歩行方策が学習した速度指令の範囲に収まっていること
    - 転倒とみなす傾きが、歩行方策の学習時より緩くないこと（それ以上傾いた状態を歩行方策は経験していない）

    Raises:
        InvalidPushConfigError: いずれかを満たさない。
    """
    if not is_integer_ratio(task.episode.high_level_period_s, gait.env.control_dt_s):
        raise InvalidPushConfigError(
            f"episode.high_level_period_s ({task.episode.high_level_period_s}) は "
            f"歩行方策の制御周期 env.control_dt_s ({gait.env.control_dt_s}) の整数倍が必要"
        )
    require_commands_within_gait_range(task.commands, gait.command)
    if task.episode.fall_roll_deg > gait.env.termination_roll_deg:
        raise InvalidPushConfigError(
            f"episode.fall_roll_deg ({task.episode.fall_roll_deg}) は歩行方策の学習時の "
            f"env.termination_roll_deg ({gait.env.termination_roll_deg}) 以下が必要"
        )
    if task.episode.fall_pitch_deg > gait.env.termination_pitch_deg:
        raise InvalidPushConfigError(
            f"episode.fall_pitch_deg ({task.episode.fall_pitch_deg}) は歩行方策の学習時の "
            f"env.termination_pitch_deg ({gait.env.termination_pitch_deg}) 以下が必要"
        )


def require_commands_within_gait_range(limits: PushCommandLimits, gait_command: VelocityCommandConfig) -> None:
    """速度指令の範囲が、歩行方策が学習した速度指令の範囲に収まっていなければ InvalidPushConfigError を送出する。"""
    gait_ranges = (gait_command.lin_vel_x_mps, gait_command.lin_vel_y_mps, gait_command.yaw_rate_radps)
    for name, limit, trained in zip(
        ("vx_mps", "vy_mps", "yaw_rate_radps"), limits.as_ranges(), gait_ranges, strict=True
    ):
        if limit.low < trained.low or limit.high > trained.high:
            raise InvalidPushConfigError(
                f"commands.{name} [{limit.low}, {limit.high}] は歩行方策が学習した範囲 "
                f"[{trained.low}, {trained.high}] に収める必要がある"
            )


def require_rule_based_within_commands(rule_based: RuleBasedPushConfig, limits: PushCommandLimits) -> None:
    """規則ベースの制御が出す速度指令が、高レベル方策の指令の範囲の外なら InvalidPushConfigError を送出する。

    範囲の外の指令は環境が切り詰めるため、設定した速さで動かないまま気づけなくなる。
    """
    forward_high = limits.vx_mps.high
    for name, speed in (
        ("approach_speed_mps", rule_based.approach_speed_mps),
        ("push_speed_mps", rule_based.push_speed_mps),
    ):
        if speed > forward_high:
            raise InvalidPushConfigError(
                f"rule_based.{name} ({speed}) は commands.vx_mps の上限 {forward_high} 以下が必要"
            )
    yaw = limits.yaw_rate_radps
    if rule_based.max_yaw_rate_radps > min(-yaw.low, yaw.high):
        raise InvalidPushConfigError(
            f"rule_based.max_yaw_rate_radps ({rule_based.max_yaw_rate_radps}) は "
            f"commands.yaw_rate_radps [{yaw.low}, {yaw.high}] に左右とも収める必要がある"
        )


@dataclass(frozen=True)
class PushTrainingRunConfig:
    """高レベル方策の学習の実行条件。

    Attributes:
        seed: 学習に使う乱数シード。評価には別のシードを使う。
        max_iterations: 学習の反復回数。正の整数。
    """

    seed: int
    max_iterations: int

    def __post_init__(self) -> None:
        _check.positive("training.max_iterations", self.max_iterations)
