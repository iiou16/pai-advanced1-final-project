"""ROS 2 の実行系で受け取る指令の有効期限。時刻はシミュレーション時刻の整数ナノ秒で扱う。

浮動小数の秒で足し引きすると、制御周期の倍数の時刻が丸め誤差で前後し、期限の境界の判定がぶれるため。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError

NANOSECONDS_PER_SECOND = 1_000_000_000


class InvalidCommandExpiryConfigError(InvalidConfigError):
    """指令の有効期限の設定が不正であることを表す。"""


def seconds_to_ns(seconds: float) -> int:
    """秒をナノ秒の整数に丸めて返す。"""
    return round(seconds * NANOSECONDS_PER_SECOND)


def ns_to_seconds(nanoseconds: int) -> float:
    """ナノ秒を秒に直して返す。"""
    return nanoseconds / NANOSECONDS_PER_SECOND


@dataclass(frozen=True)
class CommandExpiryConfig:
    """受け手が指令を古いとみなすまでの時間。超えたら停止の指令に切り替える（要件 5.5）。

    Attributes:
        velocity_command_s: 速度指令（高レベル方策 → 歩行方策）の有効期限 [s]（シミュレーション時間）。
        joint_target_s: 関節目標（歩行方策 → シミュレーション）の有効期限 [s]（シミュレーション時間）。
    """

    velocity_command_s: float
    joint_target_s: float

    def __post_init__(self) -> None:
        check = ConfigValueChecker(InvalidCommandExpiryConfigError)
        check.positive("velocity_command_s", self.velocity_command_s)
        check.positive("joint_target_s", self.joint_target_s)


def is_command_fresh(command_stamp_ns: int, now_ns: int, expiry_ns: int) -> bool:
    """時刻 command_stamp_ns の指令が、時刻 now_ns でまだ有効期限内（経過時間が expiry_ns 以下）かを返す。

    指令の時刻が now_ns より後でも有効とする。送り手が受け手より先に新しい状態を受け取って指令を決めると、
    受け手がまだ処理していない時刻の指令が届くため。
    """
    return now_ns - command_stamp_ns <= expiry_ns
