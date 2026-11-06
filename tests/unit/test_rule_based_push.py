import math

import pytest
import torch

from cooperative_go2.entities.push_config import InvalidPushConfigError, RuleBasedPushConfig
from cooperative_go2.entities.rule_based_push import (
    RuleBasedPushInputs,
    RuleBasedPushPhase,
    compute_rule_based_push_commands,
    compute_waypoint_commands,
)

CONFIG = RuleBasedPushConfig(
    approach_speed_mps=0.6,
    push_speed_mps=0.3,
    yaw_gain_per_s=2.0,
    max_yaw_rate_radps=1.0,
    standoff_m=0.8,
    side_offset_m=0.7,
    align_min_cos=0.95,
)
GOAL_RADIUS_M = 0.3
# 箱は原点、目標は +x 方向に 3 m。押す方向は +x、押し始める位置は (-0.8, 0)
BOX = (0.0, 0.0)
GOAL = (3.0, 0.0)


def _run(robot_xy: tuple[float, float], robot_yaw: float, box_xy: tuple[float, float] = BOX):
    inputs = RuleBasedPushInputs(
        robot_xy_m=torch.tensor([robot_xy]),
        robot_yaw_rad=torch.tensor([robot_yaw]),
        box_xy_m=torch.tensor([box_xy]),
        goal_xy_m=torch.tensor([GOAL]),
    )
    output = compute_rule_based_push_commands(inputs, CONFIG, GOAL_RADIUS_M)
    return output.commands[0], RuleBasedPushPhase(int(output.phases[0]))


def test_stops_when_box_is_inside_goal_region() -> None:
    commands, phase = _run((2.0, 0.0), 0.0, box_xy=(2.7, 0.0))

    assert phase == RuleBasedPushPhase.DONE
    assert commands.tolist() == [0.0, 0.0, 0.0]


def test_stops_without_nan_when_box_is_exactly_at_goal_center() -> None:
    commands, phase = _run((2.0, 0.0), 0.0, box_xy=GOAL)

    assert phase == RuleBasedPushPhase.DONE
    assert commands.tolist() == [0.0, 0.0, 0.0]


def test_pushes_straight_when_robot_is_directly_behind_box_facing_it() -> None:
    commands, phase = _run((-0.5, 0.0), 0.0)

    assert phase == RuleBasedPushPhase.PUSH
    assert torch.allclose(commands, torch.tensor([CONFIG.push_speed_mps, 0.0, 0.0]))


def test_approaches_start_point_when_behind_start_line_but_off_axis() -> None:
    # (-2, 1.5) から箱を見る向きは押す方向と大きくずれるので、まず押し始める位置 (-0.8, 0) へ向かう
    _, phase = _run((-2.0, 1.5), 0.0)

    assert phase == RuleBasedPushPhase.APPROACH


def test_approach_turns_toward_start_point() -> None:
    # 押し始める位置は右前方（ワールド -y 側）にあるので、時計回り（負のヨー角速度）に旋回する
    commands, _ = _run((-2.0, 1.5), 0.0)

    assert commands[2].item() < 0.0
    assert 0.0 < commands[0].item() < CONFIG.approach_speed_mps


def test_robot_in_front_of_box_first_steps_aside_on_its_own_side() -> None:
    # 箱の前（目標側）でわずかに左にいる。押す方向の線から離れるため、左（+y）の真横の点へ向かう
    commands, phase = _run((1.0, 0.1), 0.0)

    assert phase == RuleBasedPushPhase.GO_AROUND
    # 目標点 (1.0, 0.7) は真左なので、反時計回りに上限まで旋回し、前進はしない
    assert torch.allclose(commands, torch.tensor([0.0, 0.0, CONFIG.max_yaw_rate_radps]), atol=1e-6)


def test_robot_clear_of_push_line_goes_around_to_side_of_start_point() -> None:
    # 箱の前で右に十分離れている。押し始める位置の右横 (-0.8, -0.7) へ、ほぼ真後ろ向きに進む
    commands, phase = _run((1.0, -1.0), math.pi)

    assert phase == RuleBasedPushPhase.GO_AROUND
    heading_to_target = math.atan2(-0.7 - (-1.0), -0.8 - 1.0)
    expected_yaw_rate = CONFIG.yaw_gain_per_s * (heading_to_target - math.pi)
    assert commands[2].item() == pytest.approx(expected_yaw_rate, abs=1e-5)
    assert commands[0].item() > 0.0


def test_does_not_move_forward_when_target_is_behind_robot() -> None:
    commands, phase = _run((-2.0, 0.0), math.pi)

    assert phase == RuleBasedPushPhase.PUSH
    assert commands[0].item() == 0.0
    assert abs(commands[2].item()) == pytest.approx(CONFIG.max_yaw_rate_radps)


def test_each_env_gets_its_own_phase() -> None:
    inputs = RuleBasedPushInputs(
        robot_xy_m=torch.tensor([[-0.5, 0.0], [-2.0, 1.5], [1.0, 0.1], [2.0, 0.0]]),
        robot_yaw_rad=torch.zeros(4),
        box_xy_m=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.8, 0.0]]),
        goal_xy_m=torch.tensor([GOAL] * 4),
    )

    output = compute_rule_based_push_commands(inputs, CONFIG, GOAL_RADIUS_M)

    assert output.phases.tolist() == [
        RuleBasedPushPhase.PUSH,
        RuleBasedPushPhase.APPROACH,
        RuleBasedPushPhase.GO_AROUND,
        RuleBasedPushPhase.DONE,
    ]
    assert output.commands.shape == (4, 3)
    assert torch.all(output.commands[:, 1] == 0.0)


def test_push_direction_follows_goal_direction() -> None:
    # 目標が +y にあれば、押し始める位置は箱の -y 側。そこから +y を向いていれば押す
    inputs = RuleBasedPushInputs(
        robot_xy_m=torch.tensor([[0.0, -0.8]]),
        robot_yaw_rad=torch.tensor([math.pi / 2]),
        box_xy_m=torch.zeros(1, 2),
        goal_xy_m=torch.tensor([[0.0, 3.0]]),
    )

    output = compute_rule_based_push_commands(inputs, CONFIG, GOAL_RADIUS_M)

    assert output.phases.tolist() == [RuleBasedPushPhase.PUSH]
    assert torch.allclose(output.commands, torch.tensor([[CONFIG.push_speed_mps, 0.0, 0.0]]), atol=1e-6)


@pytest.mark.parametrize("align_min_cos", [-1.01, 1.01])
def test_config_rejects_alignment_cosine_outside_range(align_min_cos: float) -> None:
    with pytest.raises(InvalidPushConfigError, match="rule_based.align_min_cos"):
        RuleBasedPushConfig(
            approach_speed_mps=0.6,
            push_speed_mps=0.3,
            yaw_gain_per_s=2.0,
            max_yaw_rate_radps=1.0,
            standoff_m=0.8,
            side_offset_m=0.7,
            align_min_cos=align_min_cos,
        )


def test_config_rejects_non_positive_speed() -> None:
    with pytest.raises(InvalidPushConfigError, match="rule_based.push_speed_mps"):
        RuleBasedPushConfig(
            approach_speed_mps=0.6,
            push_speed_mps=0.0,
            yaw_gain_per_s=2.0,
            max_yaw_rate_radps=1.0,
            standoff_m=0.8,
            side_offset_m=0.7,
            align_min_cos=0.95,
        )


def _waypoint(robot_yaw: float, target_xy: tuple[float, float]) -> torch.Tensor:
    commands = compute_waypoint_commands(
        torch.tensor([[0.0, 0.0]]), torch.tensor([robot_yaw]), torch.tensor([target_xy]), 0.5, CONFIG
    )
    return commands[0]


def test_waypoint_commands_go_straight_when_facing_target() -> None:
    commands = _waypoint(0.0, (2.0, 0.0))

    assert commands.tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_waypoint_commands_turn_in_place_when_target_is_behind() -> None:
    commands = _waypoint(0.0, (-2.0, 0.01))

    # 後ろ向き（ずれ ≈ π）では前進せず、旋回の上限で向きを変える
    assert commands.tolist() == pytest.approx([0.0, 0.0, CONFIG.max_yaw_rate_radps])


def test_waypoint_commands_slow_down_in_proportion_to_heading_error() -> None:
    commands = _waypoint(0.0, (1.0, 1.0))

    # ずれ 45°: 前進は 0.5 × cos 45°、旋回は 2.0 × π/4 ≈ 1.57 を上限 1.0 で切る
    assert commands.tolist() == pytest.approx([0.5 * math.cos(math.pi / 4), 0.0, 1.0])
