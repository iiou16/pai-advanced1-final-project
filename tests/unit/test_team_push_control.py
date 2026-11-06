import dataclasses
import math

import pytest
import torch

from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.team_push_control import (
    InvalidTeamPushConfigError,
    TeamPushConfig,
    TeamPushInputs,
    compute_push_commands,
    compute_push_geometry,
    compute_push_predicates,
    yaw_balance_speed_scale,
)

CONFIG = TeamPushConfig(
    approach_speed_mps=0.5,
    push_speed_mps=0.6,
    yaw_gain_per_s=2.0,
    max_yaw_rate_radps=1.0,
    standoff_m=0.6,
    position_tolerance_m=0.15,
    approach_slowdown_m=0.5,
    heading_tolerance_rad=0.15,
    push_lateral_tolerance_m=0.25,
    push_heading_tolerance_rad=0.5,
    lookahead_m=0.5,
    yaw_balance_gain_per_rad=2.0,
    yaw_shift_gain_m_per_rad=0.0,
    max_yaw_shift_m=0.2,
)
# 箱は原点、退避領域は +x 方向。押す方向は +x、左（+y）側のロボットの押す位置は (-0.6, 0.25)
BOX = (0.0, 0.0)
RETREAT = (3.0, 0.0)
LEFT = 0.25


def inputs(
    robot_xy: tuple[float, float], robot_yaw: float = 0.0, box_yaw: float = 0.0, lateral_offset: float = LEFT
) -> TeamPushInputs:
    return TeamPushInputs(
        robot_xy_m=torch.tensor([robot_xy]),
        robot_yaw_rad=torch.tensor([robot_yaw]),
        box_xy_m=torch.tensor([BOX]),
        box_yaw_rad=torch.tensor([box_yaw]),
        retreat_xy_m=torch.tensor([RETREAT]),
        lateral_offset_m=lateral_offset,
    )


def test_push_position_is_behind_box_on_assigned_side() -> None:
    geometry = compute_push_geometry(inputs((-2.0, 0.0)), CONFIG)

    assert geometry.push_dir[0].tolist() == pytest.approx([1.0, 0.0])
    assert geometry.push_position_xy_m[0].tolist() == pytest.approx([-0.6, 0.25])


def test_push_position_follows_push_direction() -> None:
    # 退避領域が +y 方向なら、押す方向の左は -x。左側の押す位置は (-0.25, -0.6)
    rotated = dataclasses.replace(inputs((0.0, -2.0)), retreat_xy_m=torch.tensor([[0.0, 3.0]]))

    geometry = compute_push_geometry(rotated, CONFIG)

    assert geometry.push_position_xy_m[0].tolist() == pytest.approx([-0.25, -0.6])


def test_lateral_and_heading_errors_are_measured_from_own_line() -> None:
    geometry = compute_push_geometry(inputs((-0.6, 0.35), robot_yaw=-0.2), CONFIG)

    assert float(geometry.lateral_error_m[0]) == pytest.approx(0.1)
    assert float(geometry.heading_error_rad[0]) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("box_yaw", "expected"),
    # 箱は前後が同じ形なので、180° 回った箱は回っていないのと同じ
    [(0.1, 0.1), (-0.1, -0.1), (math.pi - 0.1, -0.1), (math.pi + 0.1, 0.1)],
    ids=["left", "right", "flipped-right", "flipped-left"],
)
def test_box_yaw_error_folds_front_and_back(box_yaw: float, expected: float) -> None:
    geometry = compute_push_geometry(inputs((-2.0, 0.0), box_yaw=box_yaw), CONFIG)

    assert float(geometry.box_yaw_error_rad[0]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("phase", "robot_xy", "robot_yaw", "expected_at", "expected_aligned"),
    [
        (TeamPhase.READY, (-0.6, 0.25), 0.0, True, True),
        (TeamPhase.READY, (-0.6, 0.25), 0.3, True, False),
        (TeamPhase.APPROACH, (-0.8, 0.25), 0.0, False, False),
        # 押す位置を越えていても、自分の線の上なら着いた（後ろ向きに戻らない）
        (TeamPhase.APPROACH, (-0.4, 0.3), 0.0, True, True),
        (TeamPhase.APPROACH, (-0.4, 0.45), 0.0, False, False),
        # 着いたあとは許容値の 2 倍（0.3 m）まで離れても外れたとみなさない
        (TeamPhase.ALIGN, (-0.85, 0.25), 0.0, True, True),
        (TeamPhase.READY, (-0.95, 0.25), 0.0, False, False),
    ],
    ids=[
        "at-and-aligned",
        "at-but-turned",
        "approach-short-of-position",
        "approach-passed-on-own-line",
        "approach-passed-off-line",
        "align-within-hold-tolerance",
        "ready-beyond-hold-tolerance",
    ],
)
def test_predicates_before_pushing(
    phase: TeamPhase, robot_xy: tuple[float, float], robot_yaw: float, expected_at: bool, expected_aligned: bool
) -> None:
    geometry = compute_push_geometry(inputs(robot_xy, robot_yaw=robot_yaw), CONFIG)

    predicates = compute_push_predicates(geometry, phase, CONFIG)

    assert bool(predicates.at_push_position[0]) is expected_at
    assert bool(predicates.aligned[0]) is expected_aligned


@pytest.mark.parametrize(
    ("robot_xy", "robot_yaw", "expected_aligned"),
    [
        # 押している間は押す位置から前へ出ていても、線と向きが許容の中なら外れたとみなさない
        ((-0.3, 0.45), 0.3, True),
        ((-0.3, 0.55), 0.0, False),
        ((-0.3, 0.25), 0.6, False),
    ],
    ids=["looser-while-pushing", "off-own-line", "turned-away"],
)
def test_predicates_while_pushing(robot_xy: tuple[float, float], robot_yaw: float, expected_aligned: bool) -> None:
    geometry = compute_push_geometry(inputs(robot_xy, robot_yaw=robot_yaw), CONFIG)

    predicates = compute_push_predicates(geometry, TeamPhase.PUSH, CONFIG)

    assert bool(predicates.at_push_position[0]) is True
    assert bool(predicates.aligned[0]) is expected_aligned


@pytest.mark.parametrize(
    "phase", [TeamPhase.INIT, TeamPhase.READY, TeamPhase.VERIFY, TeamPhase.DONE, TeamPhase.FAILED], ids=lambda p: p.name
)
def test_robot_stands_still_outside_moving_phases(phase: TeamPhase) -> None:
    robot = inputs((-2.0, 0.25))
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, phase, CONFIG)

    assert commands[0].tolist() == [0.0, 0.0, 0.0]


def test_approach_heads_to_push_position() -> None:
    robot = inputs((-2.0, 0.25))
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, TeamPhase.APPROACH, CONFIG)

    assert commands[0].tolist() == pytest.approx([CONFIG.approach_speed_mps, 0.0, 0.0])


def test_approach_slows_down_near_push_position() -> None:
    # 押す位置の 0.25 m 手前では、0.5 m の手前から比例して落とすので半分の速さ
    robot = inputs((-0.85, 0.25))
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, TeamPhase.APPROACH, CONFIG)

    assert float(commands[0, 0]) == pytest.approx(CONFIG.approach_speed_mps * 0.5)


def test_align_turns_in_place_toward_push_direction() -> None:
    robot = inputs((-0.6, 0.25), robot_yaw=-0.3)
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, TeamPhase.ALIGN, CONFIG)

    assert commands[0].tolist() == pytest.approx([0.0, 0.0, 0.6])


def test_push_goes_straight_at_push_speed_when_box_is_square() -> None:
    robot = inputs((-0.6, 0.25))
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, TeamPhase.PUSH, CONFIG)

    assert commands[0].tolist() == pytest.approx([CONFIG.push_speed_mps, 0.0, 0.0])


def test_push_steers_back_toward_own_line() -> None:
    # 線より左に 0.1 m ずれていたら右へ（ヨー角速度が負に）曲がって戻る
    robot = inputs((-0.6, 0.35))
    geometry = compute_push_geometry(robot, CONFIG)

    commands = compute_push_commands(robot, geometry, TeamPhase.PUSH, CONFIG)

    assert float(commands[0, 2]) < 0.0


def test_leading_side_slows_down_when_box_rotates() -> None:
    # 箱が左回りに 0.1 rad 回ると右側の端が前に出るので、右側（offset < 0）のロボットだけが 2 × 0.1 = 20 % 遅くなる
    left = inputs((-0.6, 0.25), box_yaw=0.1, lateral_offset=0.25)
    right = inputs((-0.6, -0.25), box_yaw=0.1, lateral_offset=-0.25)

    left_commands = compute_push_commands(left, compute_push_geometry(left, CONFIG), TeamPhase.PUSH, CONFIG)
    right_commands = compute_push_commands(right, compute_push_geometry(right, CONFIG), TeamPhase.PUSH, CONFIG)

    assert float(left_commands[0, 0]) == pytest.approx(CONFIG.push_speed_mps)
    assert float(right_commands[0, 0]) == pytest.approx(CONFIG.push_speed_mps * 0.8)


@pytest.mark.parametrize(
    ("box_yaw", "offset", "gain", "expected"),
    [(0.1, 0.0, 2.0, 1.0), (-0.1, 0.25, 2.0, 0.8), (1.0, -0.25, 2.0, 0.0), (0.1, -0.25, 0.0, 1.0)],
    ids=["single-robot-on-center-line", "left-leads", "clamped-at-zero", "balancing-disabled"],
)
def test_yaw_balance_speed_scale(box_yaw: float, offset: float, gain: float, expected: float) -> None:
    scale = yaw_balance_speed_scale(torch.tensor([box_yaw]), offset, gain)

    assert float(scale[0]) == pytest.approx(expected)


@pytest.mark.parametrize("field", ["push_speed_mps", "standoff_m", "lookahead_m", "push_lateral_tolerance_m"])
def test_config_rejects_non_positive_values(field: str) -> None:
    with pytest.raises(InvalidTeamPushConfigError):
        dataclasses.replace(CONFIG, **{field: 0.0})


def test_config_rejects_negative_yaw_balance_gain() -> None:
    with pytest.raises(InvalidTeamPushConfigError):
        dataclasses.replace(CONFIG, yaw_balance_gain_per_rad=-0.1)


def test_push_line_shifts_toward_lagging_side_when_box_rotates() -> None:
    # 箱が左回りに 0.1 rad 回ると左側の端が遅れるので、押す線を 0.5 × 0.1 = 0.05 m 左（+y）へずらす
    shifting = dataclasses.replace(CONFIG, yaw_shift_gain_m_per_rad=0.5)

    geometry = compute_push_geometry(inputs((-0.6, 0.25), box_yaw=0.1), shifting)

    assert geometry.push_position_xy_m[0].tolist() == pytest.approx([-0.6, 0.3])
    assert float(geometry.lateral_error_m[0]) == pytest.approx(-0.05)


def test_push_line_shift_is_limited() -> None:
    shifting = dataclasses.replace(CONFIG, yaw_shift_gain_m_per_rad=0.5, max_yaw_shift_m=0.1)

    geometry = compute_push_geometry(inputs((-0.6, 0.25), box_yaw=-1.0), shifting)

    assert geometry.push_position_xy_m[0].tolist() == pytest.approx([-0.6, 0.15])
