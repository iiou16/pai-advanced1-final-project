import math

import pytest
import torch

from cooperative_go2.entities.push_config import PushArenaConfig, PushDivergenceConfig
from cooperative_go2.entities.push_termination import (
    PushTerminationFlags,
    PushTerminationReason,
    classify_push_terminations,
    detect_divergence,
    is_outside_arena,
)

DIVERGENCE = PushDivergenceConfig(max_speed_mps=10.0)


def test_arena_boundary_counts_as_inside() -> None:
    arena = PushArenaConfig(half_extent_x_m=3.0, half_extent_y_m=2.0)
    positions = torch.tensor([[3.0, 2.0], [-3.0, -2.0], [3.01, 0.0], [0.0, -2.01]])

    outside = is_outside_arena(positions, arena)

    assert outside.tolist() == [False, False, True, True]


def _flags(*rows: tuple[bool, bool, bool, bool, bool]) -> PushTerminationFlags:
    columns = [torch.tensor(column) for column in zip(*rows, strict=True)]
    return PushTerminationFlags(
        time_out=columns[0], succeeded=columns[1], fallen=columns[2], out_of_bounds=columns[3], diverged=columns[4]
    )


def test_each_single_condition_maps_to_its_reason() -> None:
    flags = _flags(
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (False, False, False, True, False),
        (False, False, False, False, True),
    )

    reasons = classify_push_terminations(flags)

    assert reasons.tolist() == [
        PushTerminationReason.NONE,
        PushTerminationReason.TIME_OUT,
        PushTerminationReason.SUCCESS,
        PushTerminationReason.FALL,
        PushTerminationReason.OUT_OF_BOUNDS,
        PushTerminationReason.SIM_DIVERGED,
    ]


@pytest.mark.parametrize(
    ("flags_row", "expected"),
    [
        ((True, True, True, True, True), PushTerminationReason.SIM_DIVERGED),
        ((True, True, True, True, False), PushTerminationReason.OUT_OF_BOUNDS),
        ((True, True, True, False, False), PushTerminationReason.FALL),
        ((True, True, False, False, False), PushTerminationReason.SUCCESS),
    ],
)
def test_reason_priority_is_diverged_then_out_of_bounds_then_fall_then_success(
    flags_row: tuple[bool, bool, bool, bool, bool], expected: PushTerminationReason
) -> None:
    # 優先度の高い条件から 1 つずつ外していき、残った中で最も優先度の高い理由が選ばれることを確かめる
    reasons = classify_push_terminations(_flags(flags_row))

    assert reasons.tolist() == [expected]


def test_success_at_time_out_counts_as_success() -> None:
    reasons = classify_push_terminations(_flags((True, True, False, False, False)))

    assert reasons.tolist() == [PushTerminationReason.SUCCESS]


def test_failure_in_same_step_overrides_success() -> None:
    flags = _flags((False, True, True, False, False), (False, True, False, True, False), (True, True, True, True, True))

    reasons = classify_push_terminations(flags)

    assert reasons.tolist() == [
        PushTerminationReason.FALL,
        PushTerminationReason.OUT_OF_BOUNDS,
        PushTerminationReason.SIM_DIVERGED,
    ]


def test_divergence_is_detected_from_non_finite_state() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.3], [math.nan, 0.0, 0.3], [0.0, math.inf, 0.3]])
    yaws = torch.tensor([0.0, 0.0, 0.0])
    velocities = torch.zeros(3, 3)

    diverged = detect_divergence((positions, yaws), (velocities,), DIVERGENCE)

    assert diverged.tolist() == [False, True, True]


def test_divergence_is_detected_from_non_finite_velocity() -> None:
    velocities = torch.tensor([[0.0, 0.0], [math.nan, 0.0]])

    diverged = detect_divergence((), (velocities,), DIVERGENCE)

    # NaN の速さは上限との比較では見つからないため、有限かどうかでも調べる
    assert diverged.tolist() == [False, True]


def test_speed_above_limit_counts_as_divergence_and_limit_itself_does_not() -> None:
    robot_velocity = torch.tensor([[10.0, 0.0], [0.0, 0.0], [6.0, 8.0]])
    box_velocity = torch.tensor([[0.0, 0.0], [0.0, -10.01], [0.0, 0.0]])

    diverged = detect_divergence((), (robot_velocity, box_velocity), DIVERGENCE)

    assert diverged.tolist() == [False, True, False]


def test_divergence_requires_at_least_one_velocity() -> None:
    with pytest.raises(ValueError, match="velocities_mps"):
        detect_divergence((torch.zeros(2, 3),), (), DIVERGENCE)
