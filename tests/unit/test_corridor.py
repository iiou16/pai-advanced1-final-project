import math

import pytest
import torch

from cooperative_go2.entities.corridor import (
    AxisAlignedRect,
    BoxFootprint,
    CorridorConfig,
    InvalidCorridorConfigError,
    PassageStage,
    advance_passage_stage,
    footprint_intersects_rect,
    is_corridor_open,
)

# 通路は x ∈ [1, 2]、幅 1.2（y ∈ [-0.6, 0.6]）。通行領域は入口・出口の外へ 0.5 ずつ延ばして x ∈ [0.5, 2.5]
CORRIDOR = CorridorConfig(entrance_x_m=1.0, exit_x_m=2.0, center_y_m=0.0, width_m=1.2, apron_m=0.5)
UNIT_SQUARE = AxisAlignedRect(min_x_m=0.0, max_x_m=1.0, min_y_m=0.0, max_y_m=1.0)
SMALL_SQUARE_HALF = (0.2, 0.2)
QUARTER_SQUARE_HALF = (0.25, 0.25)
# 奥行き 0.4・幅 1.0 の箱（2 台で押す箱と同じ縦横比）
WIDE_BOX_HALF = (0.2, 0.5)
THIN_BOX_HALF = (0.6, 0.05)


def test_passage_region_extends_corridor_by_apron() -> None:
    assert CORRIDOR.passage_region() == AxisAlignedRect(min_x_m=0.5, max_x_m=2.5, min_y_m=-0.6, max_y_m=0.6)


@pytest.mark.parametrize(
    ("entrance_x", "exit_x", "width", "apron"),
    [(2.0, 2.0, 1.2, 0.5), (1.0, 2.0, 0.0, 0.5), (1.0, 2.0, 1.2, -0.1), (1.0, math.nan, 1.2, 0.5)],
    ids=["exit-not-beyond-entrance", "zero-width", "negative-apron", "nan-exit"],
)
def test_corridor_rejects_invalid_geometry(entrance_x: float, exit_x: float, width: float, apron: float) -> None:
    with pytest.raises(InvalidCorridorConfigError):
        CorridorConfig(entrance_x_m=entrance_x, exit_x_m=exit_x, center_y_m=0.0, width_m=width, apron_m=apron)


@pytest.mark.parametrize(
    ("center", "yaw_rad", "half_size", "expected"),
    [
        ((0.5, 0.5), 0.0, SMALL_SQUARE_HALF, True),
        ((3.0, 0.5), 0.0, SMALL_SQUARE_HALF, False),
        # 箱の左端 x = 1.25 - 0.25 = 1.0 が領域の右端にちょうど接する（接するだけでも重なりとみなす）。
        # float32 で丸めずに表せる寸法にする
        ((1.25, 0.5), 0.0, QUARTER_SQUARE_HALF, True),
        ((1.2501, 0.5), 0.0, QUARTER_SQUARE_HALF, False),
        # 45° 回した正方形（頂点が ±x・±y 方向、中心から 0.2√2）。軸に沿った外接の正方形は領域の角 (1, 1) と
        # 重なるが、領域の角に向いた辺 x + y = 2.5 - 0.2√2 ≈ 2.22 は角に届かない
        ((1.25, 1.25), math.pi / 4, SMALL_SQUARE_HALF, False),
        # 中心を角へ寄せると、辺 x + y = 2.2 - 0.2√2 ≈ 1.92 が角を越える
        ((1.1, 1.1), math.pi / 4, SMALL_SQUARE_HALF, True),
        # 45° 傾けた細長い箱。外接の長方形は領域と重なるが、箱の短い辺の方向への射影は離れている
        ((1.3, 1.3), -math.pi / 4, THIN_BOX_HALF, False),
        # 90° 回すと幅 1.0 の辺が x 方向になり、中心 x = 1.45 でも左端 0.95 が領域に入る
        ((1.45, 0.5), math.pi / 2, WIDE_BOX_HALF, True),
        ((1.45, 0.5), 0.0, WIDE_BOX_HALF, False),
    ],
    ids=[
        "inside",
        "far-outside",
        "touching-edge",
        "just-beyond-edge",
        "rotated-corner-misses",
        "rotated-corner-overlaps",
        "separated-on-box-axis",
        "rotated-90-overlaps",
        "unrotated-clear",
    ],
)
def test_footprint_intersects_rect(
    center: tuple[float, float], yaw_rad: float, half_size: tuple[float, float], expected: bool
) -> None:
    box = BoxFootprint(center_xy_m=torch.tensor([center]), yaw_rad=torch.tensor([yaw_rad]), half_size_xy_m=half_size)

    assert bool(footprint_intersects_rect(box, UNIT_SQUARE)[0]) is expected


def test_corridor_is_open_only_after_box_leaves_passage_region() -> None:
    box = BoxFootprint(
        center_xy_m=torch.tensor([[1.5, 0.0], [2.6, 0.0], [2.8, 0.0]]),
        yaw_rad=torch.zeros(3),
        half_size_xy_m=WIDE_BOX_HALF,
    )

    open_mask = is_corridor_open(box, CORRIDOR)

    # 通路の中、出口のすぐ外（後端 2.4 が通行領域の中）、通行領域の外（後端 2.6）
    assert open_mask.tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("stage", "robot_xy", "corridor_open", "expected"),
    [
        (PassageStage.WAITING, (0.4, 1.5), True, PassageStage.ELIGIBLE),
        (PassageStage.WAITING, (0.4, 0.0), False, PassageStage.WAITING),
        (PassageStage.WAITING, (1.5, 0.0), True, PassageStage.WAITING),
        # 開く前から通路にいて（箱を押して）出口を越えたロボットは数えない
        (PassageStage.WAITING, (2.6, 0.0), True, PassageStage.WAITING),
        # 入口側にいた時点で開いていれば、通路に入るとき・通過のときは開いているかを問わない
        (PassageStage.ELIGIBLE, (1.5, 0.0), False, PassageStage.IN_CORRIDOR),
        (PassageStage.ELIGIBLE, (1.5, 0.7), True, PassageStage.ELIGIBLE),
        # 通路に入らずに（壁の外を回り込んで）出口側へ出ても数えない
        (PassageStage.ELIGIBLE, (2.6, 0.0), True, PassageStage.ELIGIBLE),
        # 出口側の端と通路の幅の境界はどちらも含む
        (PassageStage.IN_CORRIDOR, (2.5, 0.6), False, PassageStage.PASSED),
        (PassageStage.IN_CORRIDOR, (2.6, 0.7), True, PassageStage.IN_CORRIDOR),
        (PassageStage.IN_CORRIDOR, (2.2, 0.0), True, PassageStage.IN_CORRIDOR),
        # 入口から通路の外へ戻ったら、もう一度通路に入るまで数えない
        (PassageStage.IN_CORRIDOR, (0.9, 0.0), True, PassageStage.ELIGIBLE),
        (PassageStage.PASSED, (0.0, 0.0), False, PassageStage.PASSED),
    ],
    ids=[
        "entrance-side-while-open",
        "entrance-side-while-closed",
        "inside-corridor-when-opened",
        "pusher-beyond-exit",
        "enters-corridor",
        "beside-corridor-outside-width",
        "went-around-walls",
        "crosses-exit-within-width",
        "crosses-exit-outside-width",
        "between-exit-and-region-end",
        "backs-out-of-entrance",
        "passed-is-absorbing",
    ],
)
def test_advance_passage_stage(
    stage: PassageStage, robot_xy: tuple[float, float], corridor_open: bool, expected: PassageStage
) -> None:
    advanced = advance_passage_stage(
        torch.tensor([int(stage)]), torch.tensor([robot_xy]), torch.tensor([corridor_open]), CORRIDOR
    )

    assert PassageStage(int(advanced[0])) is expected


def test_passage_stages_advance_per_environment() -> None:
    stages = torch.tensor([int(PassageStage.WAITING), int(PassageStage.IN_CORRIDOR)])
    robot_xy = torch.tensor([[0.0, 0.0], [3.0, 0.0]])

    advanced = advance_passage_stage(stages, robot_xy, torch.tensor([True, True]), CORRIDOR)

    assert advanced.tolist() == [int(PassageStage.ELIGIBLE), int(PassageStage.PASSED)]


def test_walk_from_entrance_side_through_corridor_passes() -> None:
    # 開いた通路を入口側から出口側まで 0.1 m ずつ歩くと、途中で IN_CORRIDOR を経て PASSED になる
    stage = torch.tensor([int(PassageStage.WAITING)])
    seen = []
    for step in range(31):
        robot_xy = torch.tensor([[0.0 + 0.1 * step, 0.0]])
        stage = advance_passage_stage(stage, robot_xy, torch.tensor([True]), CORRIDOR)
        seen.append(PassageStage(int(stage[0])))

    assert seen[0] is PassageStage.ELIGIBLE
    assert PassageStage.IN_CORRIDOR in seen
    assert seen[-1] is PassageStage.PASSED
