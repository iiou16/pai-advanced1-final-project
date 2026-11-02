"""1 台の箱押し（シーン S1_push）の高レベル方策に使う設定の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/push_single.yaml）にすべての値を書く。
各クラスは生成時に値の不変条件を検証し、違反すれば InvalidPushConfigError を送出する。
"""

import math
from dataclasses import dataclass, fields

# 時間の比を切り上げるときの丸め桁数。0.1 のような 2 進で割り切れない周期の丸め誤差で 1 ステップ多く数えないようにする
_RATIO_ROUND_DIGITS = 9


class InvalidPushConfigError(ValueError):
    """箱押しの設定値が不変条件（正の値、値域など）を満たさないことを表す。"""


# NaN は大小の比較がすべて偽になり「0 以下か」の検査をすり抜けるため、有限かどうかを先に調べる
def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise InvalidPushConfigError(f"{name} は有限の値が必要: {value}")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise InvalidPushConfigError(f"{name} は正の値が必要: {value}")


@dataclass(frozen=True)
class PushSuccessConfig:
    """成功判定の条件。箱の中心が目標領域に入り、その状態を hold_time_s 続けたら成功とする。

    Attributes:
        goal_radius_m: 目標領域（目標位置を中心とする円）の半径 [m]。正の値。
        hold_time_s: 目標領域に入った状態を続ける時間 [s]。正の値。
    """

    goal_radius_m: float
    hold_time_s: float

    def __post_init__(self) -> None:
        _require_positive("success.goal_radius_m", self.goal_radius_m)
        _require_positive("success.hold_time_s", self.hold_time_s)

    def required_hold_steps(self, control_dt_s: float) -> int:
        """hold_time_s を、高レベル方策の制御ステップ数に換算して返す（端数は切り上げ）。

        Args:
            control_dt_s: 高レベル方策の制御周期 [s]。正の値でなければ InvalidPushConfigError を送出する。
        """
        _require_positive("control_dt_s", control_dt_s)
        return math.ceil(round(self.hold_time_s / control_dt_s, _RATIO_ROUND_DIGITS))


@dataclass(frozen=True)
class PushArenaConfig:
    """シーンの有効範囲。環境の原点を中心とする長方形で、外に出た箱やロボットはシーン外への離脱とみなす。

    Attributes:
        half_extent_x_m: x 方向の半幅 [m]。正の値。
        half_extent_y_m: y 方向の半幅 [m]。正の値。
    """

    half_extent_x_m: float
    half_extent_y_m: float

    def __post_init__(self) -> None:
        _require_positive("arena.half_extent_x_m", self.half_extent_x_m)
        _require_positive("arena.half_extent_y_m", self.half_extent_y_m)


@dataclass(frozen=True)
class PushDivergenceConfig:
    """物理計算の発散とみなす条件。値が NaN・無限大になった環境のほか、速さが上限を超えた環境も発散とみなす。

    Attributes:
        max_speed_mps: ロボットや箱の並進の速さの上限 [m/s]。歩行や押す動作では起こりえない大きさにする。正の値。
    """

    max_speed_mps: float

    def __post_init__(self) -> None:
        _require_positive("divergence.max_speed_mps", self.max_speed_mps)


@dataclass(frozen=True)
class PushObservationScales:
    """高レベル方策の観測の各成分に掛ける正規化係数。

    Attributes:
        position: 相対位置 [m] に掛ける係数。
        lin_vel: 速度と速度指令の並進成分 [m/s] に掛ける係数。
        ang_vel: ヨー角速度と速度指令の旋回成分 [rad/s] に掛ける係数。
    """

    position: float
    lin_vel: float
    ang_vel: float

    def __post_init__(self) -> None:
        _require_positive("observation.position", self.position)
        _require_positive("observation.lin_vel", self.lin_vel)
        _require_positive("observation.ang_vel", self.ang_vel)


@dataclass(frozen=True)
class PushRewardScales:
    """報酬の各項の係数。正は報酬、負は減点。0 ならその項を使わない。フィールド名は PushRewardTerms と一致させる。"""

    box_progress: float
    success: float
    intended_face_contact: float
    box_rotation: float
    robot_fall: float
    unsafe_collision: float
    control_effort: float

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_finite(f"reward.scales.{field.name}", getattr(self, field.name))


@dataclass(frozen=True)
class PushRewardConfig:
    """報酬の計算に使うパラメータ。

    Attributes:
        intended_face_min_cos: 「意図した面を押している」とみなす、ロボット→箱の向きと箱→目標の向きの
            なす角の余弦の下限。-1 以上 1 以下。1 に近いほど、箱の真後ろから押す場合だけを認める。
        scales: 各項の係数。
    """

    intended_face_min_cos: float
    scales: PushRewardScales

    def __post_init__(self) -> None:
        if not -1.0 <= self.intended_face_min_cos <= 1.0:
            raise InvalidPushConfigError(
                f"reward.intended_face_min_cos は -1 以上 1 以下が必要: {self.intended_face_min_cos}"
            )


@dataclass(frozen=True)
class RuleBasedPushConfig:
    """規則ベースの箱押し（実験 E1 のベースライン）の制御パラメータ。

    速度と旋回の上限は、歩行方策が学習した速度指令の範囲に収める（範囲外の指令には追従が保証されないため）。

    Attributes:
        approach_speed_mps: 箱の後ろへ回り込むときの前進速度 [m/s]。正の値。
        push_speed_mps: 箱を押すときの前進速度 [m/s]。正の値。
        yaw_gain_per_s: 向きの誤差 [rad] からヨー角速度の指令 [rad/s] を作る比例ゲイン [1/s]。正の値。
        max_yaw_rate_radps: ヨー角速度の指令の絶対値の上限 [rad/s]。正の値。
        standoff_m: 押し始める位置（箱の中心から押す方向の逆へ standoff_m 下がった点）までの距離 [m]。
            箱とロボットの奥行きの半分の和より大きくし、回り込みの途中で箱に触れないようにする。正の値。
        side_offset_m: 箱の横を回り込むときに、押す方向の線から横へ離す距離 [m]。
            箱とロボットの幅の半分の和より大きくする。正の値。
        align_min_cos: 「箱の真後ろにいる」とみなす、ロボット → 箱の向きと押す方向のなす角の余弦の下限。
            -1 以上 1 以下。1 に近いほど、まっすぐ後ろにいるときだけ押し始める。
    """

    approach_speed_mps: float
    push_speed_mps: float
    yaw_gain_per_s: float
    max_yaw_rate_radps: float
    standoff_m: float
    side_offset_m: float
    align_min_cos: float

    def __post_init__(self) -> None:
        _require_positive("rule_based.approach_speed_mps", self.approach_speed_mps)
        _require_positive("rule_based.push_speed_mps", self.push_speed_mps)
        _require_positive("rule_based.yaw_gain_per_s", self.yaw_gain_per_s)
        _require_positive("rule_based.max_yaw_rate_radps", self.max_yaw_rate_radps)
        _require_positive("rule_based.standoff_m", self.standoff_m)
        _require_positive("rule_based.side_offset_m", self.side_offset_m)
        if not -1.0 <= self.align_min_cos <= 1.0:
            raise InvalidPushConfigError(f"rule_based.align_min_cos は -1 以上 1 以下が必要: {self.align_min_cos}")
