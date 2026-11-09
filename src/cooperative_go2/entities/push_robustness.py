"""1 台の箱押しの頑健性の評価で変える条件（箱の質量・摩擦・配置・速度指令の遅れ）の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/push_robustness.yaml）に条件ごとにすべての値を書く。
学習時の条件から 1 種類ずつ変えた条件を並べ、条件ごとに評価の指標を出すために使う。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker
from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.entities.push_task_config import PushResetConfig

_check = ConfigValueChecker(InvalidPushConfigError)


class UnknownRobustnessConditionError(ValueError):
    """指定した名前の条件が、頑健性の評価の設定に無いことを表す。"""


@dataclass(frozen=True)
class PushRobustnessCondition:
    """頑健性の評価の 1 条件。評価の設定（シーンと配置）のうち、ここに挙げた値だけを置き換える。

    Attributes:
        name: 条件の名前。設定の中で重複しない、空でない文字列。
        box_mass_kg: 箱の質量 [kg]。正の値。
        box_friction: 箱の表面の摩擦係数。正の値。接触する 2 面の摩擦は大きい方が使われるため、
            床の摩擦より小さくしても実際の摩擦は変わらない（評価する側で弾く）。
        reset: エピソードの始めの箱と目標の配置。
        command_delay_gait_steps: 高レベル方策の速度指令が歩行方策に届くまでの遅れ（歩行の制御周期の数）。0 以上の整数。
    """

    name: str
    box_mass_kg: float
    box_friction: float
    reset: PushResetConfig
    command_delay_gait_steps: int

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidPushConfigError("条件の name は空でない文字列が必要")
        _check.positive(f"{self.name}.box_mass_kg", self.box_mass_kg)
        _check.positive(f"{self.name}.box_friction", self.box_friction)
        delay = self.command_delay_gait_steps
        if isinstance(delay, bool) or not isinstance(delay, int):
            raise InvalidPushConfigError(f"{self.name}.command_delay_gait_steps は整数が必要: {delay!r}")
        _check.non_negative(f"{self.name}.command_delay_gait_steps", delay)


@dataclass(frozen=True)
class PushRobustnessSuite:
    """頑健性の評価の条件の一覧。

    Attributes:
        conditions: 評価する条件。1 つ以上で、名前が重複しない。
    """

    conditions: tuple[PushRobustnessCondition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise InvalidPushConfigError("conditions は 1 つ以上が必要")
        names = [condition.name for condition in self.conditions]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise InvalidPushConfigError(f"conditions の name が重複している: {duplicated}")

    def get_condition(self, name: str) -> PushRobustnessCondition:
        """名前が name の条件を返す。

        Raises:
            UnknownRobustnessConditionError: その名前の条件が無い。
        """
        for condition in self.conditions:
            if condition.name == name:
                return condition
        known = [condition.name for condition in self.conditions]
        raise UnknownRobustnessConditionError(f"条件 {name!r} は設定に無い。設定にある条件: {known}")
