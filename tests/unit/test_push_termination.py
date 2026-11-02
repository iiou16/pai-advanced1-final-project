import torch

from cooperative_go2.entities.push_config import PushArenaConfig
from cooperative_go2.entities.push_termination import (
    PushTerminationFlags,
    PushTerminationReason,
    classify_push_terminations,
    is_outside_arena,
)


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
