"""通路のシーン（S2_corridor・S3_team）の設定の値オブジェクト。

壁で挟んだ通路をふさぐ箱を、通路の出口の先の退避領域へ押し出して、通路を開通させるシーン。
既定値はここに持たず、設定ファイル（configs/scenes/s2_corridor.yaml など）にすべての値を書く。
床とロボットの物理条件（制御周期・PD ゲイン・初期の高さと姿勢・摩擦）は、歩行方策を学習したときの条件を
そのまま使うため、ここには持たない。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError
from cooperative_go2.entities.corridor import BoxFootprint, CorridorConfig, footprint_intersects_rect
from cooperative_go2.entities.push_scene_config import PLANAR_POSITION_ELEMENTS, BoxConfig



class InvalidCorridorSceneConfigError(InvalidConfigError):
    """通路のシーンの設定値が不変条件（箱が通路をふさいでいる、退避領域が通行領域の外にある等）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidCorridorSceneConfigError)


@dataclass(frozen=True)
class WallConfig:
    """通路の両側の壁。通路の入口から出口まで、通路の縁から y 方向に outer_half_extent_y_m まで延びる固定の直方体。

    Attributes:
        height_m: 壁の高さ [m]。正の値。Go2 が乗り越えられない高さにする。
        outer_half_extent_y_m: 壁の外側の端の、通路の中心線からの距離 [m]。通路の幅の半分より大きい。
            シーンの有効範囲の端まで延ばし、Go2 が壁を回り込めないようにする。
    """

    height_m: float
    outer_half_extent_y_m: float

    def __post_init__(self) -> None:
        _check.positive("walls.height_m", self.height_m)
        _check.positive("walls.outer_half_extent_y_m", self.outer_half_extent_y_m)


@dataclass(frozen=True)
class RetreatAreaConfig:
    """箱を押し出す先の退避領域。中心と半径の円で、ClearCorridor の目標として依頼する。

    Attributes:
        center_xy_m: 退避領域の中心 (x, y) [m]。
        radius_m: 退避領域の半径 [m]。正の値。
    """

    center_xy_m: tuple[float, ...]
    radius_m: float

    def __post_init__(self) -> None:
        if len(self.center_xy_m) != PLANAR_POSITION_ELEMENTS:
            raise InvalidCorridorSceneConfigError(f"retreat_area.center_xy_m は 2 要素が必要: {self.center_xy_m}")
        for axis, coordinate in zip("xy", self.center_xy_m, strict=True):
            _check.finite(f"retreat_area.center_xy_m.{axis}", coordinate)
        _check.positive("retreat_area.radius_m", self.radius_m)


@dataclass(frozen=True)
class RobotPlacement:
    """1 台の Go2 の名前と初期位置。向きは歩行方策の学習時と同じ（+x 方向）にする。

    Attributes:
        name: ロボットの名前（ROS 2 の Namespace にも使う。例: go2_a）。空でない。
        initial_xy_m: 胴体の初期位置 (x, y) [m]。高さは歩行方策の学習時の初期位置の高さを使う。
    """

    name: str
    initial_xy_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidCorridorSceneConfigError("robots[].name は空にできない")
        if len(self.initial_xy_m) != PLANAR_POSITION_ELEMENTS:
            raise InvalidCorridorSceneConfigError(f"robots[{self.name}].initial_xy_m は 2 要素が必要")
        for axis, coordinate in zip("xy", self.initial_xy_m, strict=True):
            _check.finite(f"robots[{self.name}].initial_xy_m.{axis}", coordinate)


@dataclass(frozen=True)
class CorridorSceneConfig:
    """通路のシーンの配置一式。

    Attributes:
        box: 通路をふさぐ箱。初期位置で箱が通行領域と重なる（通路が閉じている）こと。箱は回転させずに置く。
        corridor: 通路の形と通行領域。
        walls: 通路の両側の壁。
        retreat_area: 箱を押し出す先。円全体が通行領域の外にあること（退避領域に入れても通路が閉じたままにならない）。
        robots: 置く Go2 の名前と初期位置。1 台以上で、名前は重複しない。
    """

    box: BoxConfig
    corridor: CorridorConfig
    walls: WallConfig
    retreat_area: RetreatAreaConfig
    robots: tuple[RobotPlacement, ...]

    def __post_init__(self) -> None:
        if self.walls.outer_half_extent_y_m <= self.corridor.width_m / 2:
            raise InvalidCorridorSceneConfigError(
                f"walls.outer_half_extent_y_m ({self.walls.outer_half_extent_y_m}) は通路の幅の半分より大きくする"
            )
        if not self.robots:
            raise InvalidCorridorSceneConfigError("robots には 1 台以上が必要")
        names = [robot.name for robot in self.robots]
        if len(set(names)) != len(names):
            raise InvalidCorridorSceneConfigError(f"robots の名前が重複している: {names}")
        if not footprint_intersects_rect(self.initial_box_footprint(), self.corridor.passage_region()).all():
            raise InvalidCorridorSceneConfigError("箱の初期位置が通行領域と重ならない（通路が最初から開いている）")
        self._require_retreat_area_outside_passage_region()

    def initial_box_footprint(self) -> BoxFootprint:
        """初期位置（回転なし）の箱を真上から見た長方形（並列環境 1 つ分）を返す。"""
        size_x, size_y, _ = self.box.size_m
        return BoxFootprint(
            center_xy_m=torch.tensor([self.box.initial_xy_m]),
            yaw_rad=torch.zeros(1),
            half_size_xy_m=(size_x / 2, size_y / 2),
        )

    def _require_retreat_area_outside_passage_region(self) -> None:
        region = self.corridor.passage_region()
        center_x, center_y = self.retreat_area.center_xy_m
        # 円の中心から通行領域（長方形）までの最短距離が半径より大きければ、円全体が外にある
        nearest_x = min(max(center_x, region.min_x_m), region.max_x_m)
        nearest_y = min(max(center_y, region.min_y_m), region.max_y_m)
        distance = ((center_x - nearest_x) ** 2 + (center_y - nearest_y) ** 2) ** 0.5
        if distance <= self.retreat_area.radius_m:
            raise InvalidCorridorSceneConfigError(
                f"退避領域が通行領域と重なる（通行領域までの距離 {distance:.3f} m ≤ 半径 {self.retreat_area.radius_m} m）"
            )
