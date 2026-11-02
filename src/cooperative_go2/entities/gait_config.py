"""歩行方策の学習・評価に使う設定の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/gait.yaml）にすべての値を書く。
各クラスは生成時に値の不変条件を検証し、違反すれば InvalidGaitConfigError を送出する。
学習アルゴリズム（PPO）固有の設定はここに置かない。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import (
    ConfigValueChecker,
    InvalidConfigError,
    ceil_ratio,
    is_integer_ratio,
)
from cooperative_go2.entities.go2 import Go2JointAngles

QUATERNION_SIZE = 4
POSITION_SIZE = 3


class InvalidGaitConfigError(InvalidConfigError):
    """歩行の設定値が不変条件（有限の値、正の値、範囲の上下関係、要素数など）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidGaitConfigError)


@dataclass(frozen=True)
class ValueRange:
    """一様分布で値を引く閉区間 [low, high]。low == high なら固定値を表す。"""

    low: float
    high: float

    def __post_init__(self) -> None:
        _check.finite("範囲の下限", self.low)
        _check.finite("範囲の上限", self.high)
        if self.low > self.high:
            raise InvalidGaitConfigError(f"範囲の下限が上限より大きい: low={self.low} high={self.high}")


@dataclass(frozen=True)
class VelocityCommandConfig:
    """学習中に引く速度指令の範囲と、引き直す周期。

    範囲から一様に引くだけでは厳密な 0 がほぼ出ず、停止やその場旋回を学習できない。そのため線速度 (vx, vy) と
    ヨー角速度を、それぞれ独立に指定の確率で 0 にする。両方が 0 になれば停止、線速度だけが 0 ならその場旋回になる。

    Attributes:
        lin_vel_x_mps: 本体座標系の前後方向の速度 [m/s]。前が正。
        lin_vel_y_mps: 本体座標系の左右方向の速度 [m/s]。左が正。
        yaw_rate_radps: ヨー角速度 [rad/s]。上から見て反時計回りが正。
        zero_lin_vel_probability: 引いた指令の vx と vy をともに 0 にする確率。0 以上 1 以下。
        zero_yaw_rate_probability: 引いた指令のヨー角速度を 0 にする確率。0 以上 1 以下。
        resampling_time_s: 指令を引き直す周期 [s]。正の値。
    """

    lin_vel_x_mps: ValueRange
    lin_vel_y_mps: ValueRange
    yaw_rate_radps: ValueRange
    zero_lin_vel_probability: float
    zero_yaw_rate_probability: float
    resampling_time_s: float

    def __post_init__(self) -> None:
        _check.unit_interval("command.zero_lin_vel_probability", self.zero_lin_vel_probability)
        _check.unit_interval("command.zero_yaw_rate_probability", self.zero_yaw_rate_probability)
        _check.positive("command.resampling_time_s", self.resampling_time_s)


@dataclass(frozen=True)
class Go2RobotConfig:
    """Go2 の既定姿勢・PD ゲイン・初期位置。

    Attributes:
        default_joint_angles_rad: 既定姿勢の関節角。行動（残差）の基準と、リセット時の関節角の基準（ここに
            env.reset_joint_pos_noise_rad のノイズを足す）に使う。
        kp: 関節の PD 制御の比例ゲイン。正の値。
        kd: 関節の PD 制御の微分ゲイン。0 以上。
        base_init_pos_m: リセット時の胴体の位置 (x, y, z) [m]。
        base_init_quat_wxyz: リセット時の胴体の姿勢（四元数 w, x, y, z）。
    """

    default_joint_angles_rad: Go2JointAngles
    kp: float
    kd: float
    base_init_pos_m: tuple[float, ...]
    base_init_quat_wxyz: tuple[float, ...]

    def __post_init__(self) -> None:
        _check.positive("robot.kp", self.kp)
        _check.non_negative("robot.kd", self.kd)
        if len(self.base_init_pos_m) != POSITION_SIZE:
            raise InvalidGaitConfigError(f"robot.base_init_pos_m は 3 要素が必要: {self.base_init_pos_m}")
        if len(self.base_init_quat_wxyz) != QUATERNION_SIZE:
            raise InvalidGaitConfigError(f"robot.base_init_quat_wxyz は 4 要素が必要: {self.base_init_quat_wxyz}")


@dataclass(frozen=True)
class GaitEnvConfig:
    """歩行学習の環境（シーン S0_flat）の設定。

    Attributes:
        num_envs: 並列環境の数。正の整数。
        control_dt_s: 方策の制御周期 = 物理ステップの周期 [s]。
        substeps: 1 制御周期あたりの物理計算の分割数。
        episode_length_s: 1 エピソードの長さ [s]。
        action_scale: 行動（既定姿勢からの残差）を関節角 [rad] に変換する係数。
        clip_actions: 行動の絶対値の上限。
        simulate_action_latency: True なら 1 制御周期前の行動を適用し、実機の 1 ステップ遅延を模擬する。
        termination_roll_deg: ロールの絶対値がこれを超えたら転倒としてエピソードを終える [deg]。
        termination_pitch_deg: ピッチの絶対値がこれを超えたら転倒としてエピソードを終える [deg]。
        max_collision_pairs: 物理エンジンが扱う衝突候補の上限。超えたら設定の不備として学習・評価を止める。
        reset_joint_pos_noise_rad: リセット時に既定姿勢の各関節角へ足す一様ノイズの幅 [rad]。
            [-幅, 幅] から関節ごとに独立に引く。0 以上。0 なら全環境が同じ初期状態から始まる。
    """

    num_envs: int
    control_dt_s: float
    substeps: int
    episode_length_s: float
    action_scale: float
    clip_actions: float
    simulate_action_latency: bool
    termination_roll_deg: float
    termination_pitch_deg: float
    max_collision_pairs: int
    reset_joint_pos_noise_rad: float

    def __post_init__(self) -> None:
        _check.positive("env.num_envs", self.num_envs)
        _check.positive("env.control_dt_s", self.control_dt_s)
        _check.positive("env.substeps", self.substeps)
        _check.positive("env.episode_length_s", self.episode_length_s)
        _check.positive("env.action_scale", self.action_scale)
        _check.positive("env.clip_actions", self.clip_actions)
        _check.positive("env.termination_roll_deg", self.termination_roll_deg)
        _check.positive("env.termination_pitch_deg", self.termination_pitch_deg)
        _check.positive("env.max_collision_pairs", self.max_collision_pairs)
        _check.non_negative("env.reset_joint_pos_noise_rad", self.reset_joint_pos_noise_rad)


@dataclass(frozen=True)
class GaitObservationScales:
    """観測の各成分に掛ける正規化係数。"""

    lin_vel: float
    ang_vel: float
    dof_pos: float
    dof_vel: float

    def __post_init__(self) -> None:
        _check.positive("observation.lin_vel", self.lin_vel)
        _check.positive("observation.ang_vel", self.ang_vel)
        _check.positive("observation.dof_pos", self.dof_pos)
        _check.positive("observation.dof_vel", self.dof_vel)


@dataclass(frozen=True)
class GaitRewardScales:
    """報酬の各項の係数。正は報酬、負は減点。0 ならその項は報酬に寄与しない（ログには出る）。

    実際の報酬は「項の値 × 係数 × 制御周期」で、制御周期を変えても 1 秒あたりの重みが変わらないようにする。
    フィールド名は GaitRewardTerms と一致させる。
    """

    tracking_lin_vel: float
    tracking_ang_vel: float
    lin_vel_z: float
    base_height: float
    action_rate: float
    similar_to_default: float


@dataclass(frozen=True)
class GaitRewardConfig:
    """報酬の計算に使うパラメータ。

    Attributes:
        tracking_sigma: 速度追従の報酬 exp(-誤差^2 / sigma) の幅。正の値。
        base_height_target_m: 胴体の高さの目標 [m]。
        scales: 各項の係数。
    """

    tracking_sigma: float
    base_height_target_m: float
    scales: GaitRewardScales

    def __post_init__(self) -> None:
        _check.positive("reward.tracking_sigma", self.tracking_sigma)
        _check.positive("reward.base_height_target_m", self.base_height_target_m)


@dataclass(frozen=True)
class GaitTrainingConfig:
    """学習の実行条件。

    Attributes:
        seed: 学習に使う乱数シード。評価には別のシードを使う。
        max_iterations: 学習の反復回数。正の整数。
    """

    seed: int
    max_iterations: int

    def __post_init__(self) -> None:
        _check.positive("training.max_iterations", self.max_iterations)


@dataclass(frozen=True)
class GaitConfig:
    """歩行方策の学習・評価に必要な、学習アルゴリズムによらない設定一式。"""

    env: GaitEnvConfig
    robot: Go2RobotConfig
    command: VelocityCommandConfig
    observation: GaitObservationScales
    reward: GaitRewardConfig
    training: GaitTrainingConfig

    def __post_init__(self) -> None:
        if not is_integer_ratio(self.command.resampling_time_s, self.env.control_dt_s):
            raise InvalidGaitConfigError(
                f"command.resampling_time_s ({self.command.resampling_time_s}) は "
                f"env.control_dt_s ({self.env.control_dt_s}) の整数倍が必要"
            )

    def command_resampling_interval_steps(self) -> int:
        """速度指令を引き直す間隔を制御ステップ数で返す。"""
        return round(self.command.resampling_time_s / self.env.control_dt_s)

    def max_episode_length_steps(self) -> int:
        """1 エピソードの長さを制御ステップ数で返す（端数は切り上げ）。"""
        return ceil_ratio(self.env.episode_length_s, self.env.control_dt_s)

