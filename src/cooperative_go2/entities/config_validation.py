"""設定値の不変条件の検査と、時間の比をステップ数に換算する計算。

歩行・箱押し・学習ライブラリの設定で共通に使う。違反は設定ごとのドメイン例外で知らせ、
そのドメイン例外はすべて InvalidConfigError を継承する（設定の読み込み側が 1 つの型で受けられるようにするため）。
"""

import math
from dataclasses import dataclass

# 周期の比が整数かを判定するときの許容誤差。0.02 のような 2 進で割り切れない値の丸め誤差を吸収する
INTEGER_RATIO_TOLERANCE = 1e-9


class InvalidConfigError(ValueError):
    """設定値が不変条件（有限の値、正の値、値域など）を満たさないことを表す。設定ごとのドメイン例外の基底。"""


@dataclass(frozen=True)
class ConfigValueChecker:
    """設定値の不変条件を検査し、違反を error_type の例外で送出する。

    NaN は大小の比較がすべて偽になり、正の値や値域の検査をすり抜けるため、どの検査も先に有限かどうかを調べる。

    Attributes:
        error_type: 違反を知らせる例外の型。InvalidConfigError を継承した、設定ごとのドメイン例外。
    """

    error_type: type[InvalidConfigError]

    def finite(self, name: str, value: float) -> None:
        """value が有限でなければ error_type を送出する。name はエラーメッセージに出す設定のキー。"""
        if not math.isfinite(value):
            raise self.error_type(f"{name} は有限の値が必要: {value}")

    def positive(self, name: str, value: float) -> None:
        """value が正の有限値でなければ error_type を送出する。"""
        self.finite(name, value)
        if value <= 0:
            raise self.error_type(f"{name} は正の値が必要: {value}")

    def non_negative(self, name: str, value: float) -> None:
        """value が 0 以上の有限値でなければ error_type を送出する。"""
        self.finite(name, value)
        if value < 0:
            raise self.error_type(f"{name} は 0 以上が必要: {value}")

    def unit_interval(self, name: str, value: float) -> None:
        """value が [0, 1] の有限値でなければ error_type を送出する。"""
        self.finite(name, value)
        if not 0 <= value <= 1:
            raise self.error_type(f"{name} は [0, 1] が必要: {value}")


def is_integer_ratio(numerator: float, denominator: float) -> bool:
    """numerator が denominator の整数倍か（丸め誤差は INTEGER_RATIO_TOLERANCE まで許す）を返す。"""
    ratio = numerator / denominator
    return abs(ratio - round(ratio)) <= INTEGER_RATIO_TOLERANCE


def ceil_ratio(numerator: float, denominator: float) -> int:
    """numerator / denominator を切り上げた整数を返す。時間の長さを制御ステップ数に換算するのに使う。

    割り切れるはずの比が丸め誤差で整数をわずかに超えても、1 ステップ多く数えない。

    Examples:
        ceil_ratio(0.3, 0.1) は 3（浮動小数点では 0.3 / 0.1 = 2.9999999999999996）、ceil_ratio(0.25, 0.1) は 3。
    """
    ratio = numerator / denominator
    rounded = round(ratio)
    if abs(ratio - rounded) <= INTEGER_RATIO_TOLERANCE:
        return rounded
    return math.ceil(ratio)
