"""通路の形と、通路の開通（箱が通行領域から出たか）・Go2 の通過の判定。

通路は x 軸に沿ってまっすぐ延び、入口（x の小さい側）から出口（x の大きい側）へ通る。
通行領域は、通路の長方形を入口と出口の外へ apron_m ずつ延ばした長方形とする。通路の長方形だけにすると、
出口をふさぐように箱が出口のすぐ外で止まっていても開通と判定してしまうため。

判定はどれもワールド座標系の水平面で行い、並列環境のバッチのまま計算する。
状態を持つ判定（通過）は、前のステップの値から次の値を計算する純粋な関数で扱う。
リセットや並列環境の扱いを呼び出し元に任せるため。
"""

from dataclasses import dataclass
from enum import IntEnum

import torch

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError

X_AXIS, Y_AXIS = 0, 1


class InvalidCorridorConfigError(InvalidConfigError):
    """通路の設定値が不変条件（正の幅、入口より先にある出口など）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidCorridorConfigError)


@dataclass(frozen=True)
class CorridorConfig:
    """x 軸に沿った、両側を壁で挟まれた通路。

    Attributes:
        entrance_x_m: 通路の入口（壁の入口側の端）の x [m]。
        exit_x_m: 通路の出口（壁の出口側の端）の x [m]。entrance_x_m より大きい。
        center_y_m: 通路の中心線の y [m]。
        width_m: 通路の幅（両側の壁の間）[m]。正の値。
        apron_m: 通行領域を通路の入口・出口の外へ延ばす長さ [m]。0 以上。
    """

    entrance_x_m: float
    exit_x_m: float
    center_y_m: float
    width_m: float
    apron_m: float

    def __post_init__(self) -> None:
        _check.finite("corridor.entrance_x_m", self.entrance_x_m)
        _check.finite("corridor.exit_x_m", self.exit_x_m)
        _check.finite("corridor.center_y_m", self.center_y_m)
        _check.positive("corridor.width_m", self.width_m)
        _check.non_negative("corridor.apron_m", self.apron_m)
        if self.exit_x_m <= self.entrance_x_m:
            raise InvalidCorridorConfigError(
                f"corridor.exit_x_m ({self.exit_x_m}) は entrance_x_m ({self.entrance_x_m}) より大きくする"
            )

    def passage_region(self) -> "AxisAlignedRect":
        """通行領域（通路を入口・出口の外へ apron_m ずつ延ばした長方形）を返す。"""
        half_width = self.width_m / 2
        return AxisAlignedRect(
            min_x_m=self.entrance_x_m - self.apron_m,
            max_x_m=self.exit_x_m + self.apron_m,
            min_y_m=self.center_y_m - half_width,
            max_y_m=self.center_y_m + half_width,
        )


@dataclass(frozen=True)
class AxisAlignedRect:
    """座標軸に沿った長方形の領域（境界を含む）。

    Attributes:
        min_x_m: x の下端 [m]。
        max_x_m: x の上端 [m]。min_x_m より大きい。
        min_y_m: y の下端 [m]。
        max_y_m: y の上端 [m]。min_y_m より大きい。
    """

    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float

    def __post_init__(self) -> None:
        if not (self.min_x_m < self.max_x_m and self.min_y_m < self.max_y_m):
            raise ValueError(f"長方形の上端は下端より大きくする: {self}")


@dataclass(frozen=True)
class BoxFootprint:
    """箱を真上から見た長方形。テンソルは並列環境のバッチ。

    Attributes:
        center_xy_m: 箱の中心 (num_envs, 2)、ワールド座標系 [m]。
        yaw_rad: 箱のヨー角 (num_envs,) [rad]。箱の x 軸（size_m の 1 つ目の辺）のワールドの x 軸からの角度。
        half_size_xy_m: 箱の x 軸・y 軸方向の寸法の半分 (x, y) [m]。全環境で共通。
    """

    center_xy_m: torch.Tensor
    yaw_rad: torch.Tensor
    half_size_xy_m: tuple[float, float]


def footprint_intersects_rect(footprint: BoxFootprint, rect: AxisAlignedRect) -> torch.Tensor:
    """箱の長方形と領域が重なる（境界が接するだけの場合を含む）環境のマスク (num_envs,) を返す。

    分離軸の定理で判定する。2 つの長方形は、領域の 2 軸と箱の 2 軸のいずれかへの射影が離れていれば重ならない。
    接するだけを重なりに含めるのは、開通の判定を安全側（箱が残っているとみなす側）に倒すため。
    """
    box_half_x, box_half_y = footprint.half_size_xy_m
    rect_half_x = (rect.max_x_m - rect.min_x_m) / 2
    rect_half_y = (rect.max_y_m - rect.min_y_m) / 2
    rect_center = footprint.center_xy_m.new_tensor(
        ((rect.min_x_m + rect.max_x_m) / 2, (rect.min_y_m + rect.max_y_m) / 2)
    )
    offset = footprint.center_xy_m - rect_center
    cos_yaw = torch.cos(footprint.yaw_rad).abs()
    sin_yaw = torch.sin(footprint.yaw_rad).abs()
    # 領域の軸（ワールドの x 軸・y 軸）に射影した箱の半分の長さ
    box_extent_x = cos_yaw * box_half_x + sin_yaw * box_half_y
    box_extent_y = sin_yaw * box_half_x + cos_yaw * box_half_y
    # 箱の軸に射影した領域の半分の長さと、中心どうしの距離
    rect_extent_box_x = cos_yaw * rect_half_x + sin_yaw * rect_half_y
    rect_extent_box_y = sin_yaw * rect_half_x + cos_yaw * rect_half_y
    box_axis_x = torch.stack((torch.cos(footprint.yaw_rad), torch.sin(footprint.yaw_rad)), dim=1)
    box_axis_y = torch.stack((-torch.sin(footprint.yaw_rad), torch.cos(footprint.yaw_rad)), dim=1)
    separated = (
        (offset[:, X_AXIS].abs() > box_extent_x + rect_half_x)
        | (offset[:, Y_AXIS].abs() > box_extent_y + rect_half_y)
        | (torch.sum(offset * box_axis_x, dim=1).abs() > box_half_x + rect_extent_box_x)
        | (torch.sum(offset * box_axis_y, dim=1).abs() > box_half_y + rect_extent_box_y)
    )
    return ~separated


def is_corridor_open(footprint: BoxFootprint, corridor: CorridorConfig) -> torch.Tensor:
    """箱が通行領域と重ならない（通路が開いている）環境のマスク (num_envs,) を返す。"""
    return ~footprint_intersects_rect(footprint, corridor.passage_region())


class PassageStage(IntEnum):
    """Go2 の通過の判定の段階。PASSED になったら、そのエピソードの間は戻らない。

    - WAITING: 通路が開いた状態で入口側（通行領域の入口側の端以下の x）にいたことがまだない。
    - ELIGIBLE: 通路が開いた状態で入口側にいた。ここから通路（両側の壁の間）に入れば IN_CORRIDOR になる。
    - IN_CORRIDOR: 入口側から通路に入った。ここから通路の幅の中で通行領域の出口側の端を越えたら通過とする。
    - PASSED: 通過した。
    """

    WAITING = 0
    ELIGIBLE = 1
    IN_CORRIDOR = 2
    PASSED = 3


def advance_passage_stage(
    stages: torch.Tensor, robot_xy_m: torch.Tensor, corridor_open: torch.Tensor, corridor: CorridorConfig
) -> torch.Tensor:
    """1 ステップ進めた後の通過の判定の段階 (num_envs,) を返す。

    「通路が開いたあとに入口から通路の中を通って出口まで通った」ことを、次の段で判定する。
    通路が開く前から通路の中にいたロボット（箱を押していたロボットなど）は、いったん入口側へ戻るまで数えない。
    開通と通過を別の指標として数えるため。通路の中を通ったことを確かめるのは、壁の外を回り込んで
    出口側へ出る経路を通過に数えないため。

    1. 通路が開いている間に、胴体の x が通行領域の入口側の端以下になったら ELIGIBLE にする。y は問わない。
    2. ELIGIBLE のロボットの胴体が通路（x が入口から出口の間、y が通路の幅の中）に入ったら IN_CORRIDOR にする。
    3. IN_CORRIDOR のロボットが通路の入口より手前へ戻ったら ELIGIBLE に戻す（入口から出て壁の外を回り込ませないため）。
    4. IN_CORRIDOR のロボットの胴体の x が通行領域の出口側の端以上になり、そのとき y が通路の幅に収まっていれば
       PASSED にする。2〜4 では通路が開いている必要はない（入口側にいた時点で開いていれば、通っている途中で
       箱が戻ってきても通過を妨げない）。

    Args:
        stages: 前のステップの段階 (num_envs,)。PassageStage の値の整数型。エピソードの最初は WAITING。
        robot_xy_m: Go2 の胴体の位置 (num_envs, 2)、ワールド座標系 [m]。
        corridor_open: このステップで通路が開いている環境のマスク (num_envs,)。is_corridor_open() の結果。
        corridor: 通路の形。
    """
    region = corridor.passage_region()
    robot_x = robot_xy_m[:, X_AXIS]
    robot_y = robot_xy_m[:, Y_AXIS]
    within_width = (robot_y >= region.min_y_m) & (robot_y <= region.max_y_m)
    on_entrance_side = robot_x <= region.min_x_m
    inside_corridor = (robot_x >= corridor.entrance_x_m) & (robot_x <= corridor.exit_x_m) & within_width
    before_entrance = robot_x < corridor.entrance_x_m
    beyond_exit = (robot_x >= region.max_x_m) & within_width
    transitions = (
        ((stages == PassageStage.WAITING) & corridor_open & on_entrance_side, PassageStage.ELIGIBLE),
        ((stages == PassageStage.ELIGIBLE) & inside_corridor, PassageStage.IN_CORRIDOR),
        ((stages == PassageStage.IN_CORRIDOR) & before_entrance, PassageStage.ELIGIBLE),
        ((stages == PassageStage.IN_CORRIDOR) & beyond_exit, PassageStage.PASSED),
    )
    # 遷移の条件は元の段ごとに分かれているので、1 ステップで 2 段進むことはない
    advanced = stages
    for condition, next_stage in transitions:
        advanced = torch.where(condition, int(next_stage), advanced)
    return advanced
