"""1 台の箱押しのシーン（S1_push）に置く箱の設定の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/scenes/s1_push.yaml）にすべての値を書く。
床とロボットの物理条件（制御周期・PD ゲイン・初期姿勢・摩擦）は、歩行方策を学習したときの条件を
そのまま使うため、ここには持たない。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError

BOX_SIZE_ELEMENTS = 3
PLANAR_POSITION_ELEMENTS = 2


class InvalidPushSceneConfigError(InvalidConfigError):
    """箱押しのシーンの設定値が不変条件（正の値、要素数など）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidPushSceneConfigError)


@dataclass(frozen=True)
class BoxConfig:
    """押す対象の箱。床の上を滑る動的な剛体として置く。

    Attributes:
        size_m: 箱の寸法 (x, y, z) [m]。各要素は正の値。
        mass_kg: 箱の質量 [kg]。正の値。
        friction: 箱の表面の摩擦係数。正の値。接触する 2 面の摩擦は大きい方が使われるため、
            箱と床の間の実際の摩擦は max(friction, 床の摩擦) になる（床の摩擦より小さくしても効かない）。
        initial_xy_m: シーンを作るときの箱の中心の位置 (x, y) [m]。各環境の原点から見た座標。
    """

    size_m: tuple[float, ...]
    mass_kg: float
    friction: float
    initial_xy_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.size_m) != BOX_SIZE_ELEMENTS:
            raise InvalidPushSceneConfigError(f"box.size_m は 3 要素が必要: {self.size_m}")
        for axis, length in zip("xyz", self.size_m, strict=True):
            _check.positive(f"box.size_m.{axis}", length)
        _check.positive("box.mass_kg", self.mass_kg)
        _check.positive("box.friction", self.friction)
        if len(self.initial_xy_m) != PLANAR_POSITION_ELEMENTS:
            raise InvalidPushSceneConfigError(f"box.initial_xy_m は 2 要素が必要: {self.initial_xy_m}")
        for axis, coordinate in zip("xy", self.initial_xy_m, strict=True):
            _check.finite(f"box.initial_xy_m.{axis}", coordinate)

    def volume_m3(self) -> float:
        """箱の体積 [m^3] を返す。"""
        size_x, size_y, size_z = self.size_m
        return size_x * size_y * size_z

    def density_kg_per_m3(self) -> float:
        """mass_kg になる一様な密度 [kg/m^3] を返す。"""
        return self.mass_kg / self.volume_m3()
