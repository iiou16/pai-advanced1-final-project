import dataclasses
import math
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig, ValueRange
from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.entities.push_layout import sample_push_layout
from cooperative_go2.entities.push_task_config import PushResetConfig, PushTaskConfig, require_compatible_with_gait
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_training_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gait() -> GaitConfig:
    return load_gait_settings(REPO_ROOT / "configs" / "gait.yaml").gait


@pytest.fixture
def task() -> PushTaskConfig:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single.yaml").task


def test_repository_push_config_is_compatible_with_repository_gait_config(
    task: PushTaskConfig, gait: GaitConfig
) -> None:
    require_compatible_with_gait(task, gait)

    # 10 Hz の高レベル方策の 1 ステップで、50 Hz の歩行方策を 5 回呼ぶ
    assert task.gait_steps_per_high_level_step(gait.env.control_dt_s) == 5


def test_high_level_period_must_be_multiple_of_gait_period(task: PushTaskConfig, gait: GaitConfig) -> None:
    episode = dataclasses.replace(task.episode, high_level_period_s=0.03)

    with pytest.raises(InvalidPushConfigError, match="high_level_period_s"):
        require_compatible_with_gait(dataclasses.replace(task, episode=episode), gait)


def test_command_outside_trained_range_is_rejected(task: PushTaskConfig, gait: GaitConfig) -> None:
    too_fast = ValueRange(low=gait.command.lin_vel_x_mps.low, high=gait.command.lin_vel_x_mps.high + 0.1)
    commands = dataclasses.replace(task.commands, vx_mps=too_fast)

    with pytest.raises(InvalidPushConfigError, match="commands.vx_mps"):
        require_compatible_with_gait(dataclasses.replace(task, commands=commands), gait)


def test_fall_threshold_looser_than_gait_training_is_rejected(task: PushTaskConfig, gait: GaitConfig) -> None:
    episode = dataclasses.replace(task.episode, fall_pitch_deg=gait.env.termination_pitch_deg + 1.0)

    with pytest.raises(InvalidPushConfigError, match="fall_pitch_deg"):
        require_compatible_with_gait(dataclasses.replace(task, episode=episode), gait)


def test_required_hold_steps_use_high_level_period(task: PushTaskConfig) -> None:
    success = dataclasses.replace(task.success, hold_time_s=1.0)
    episode = dataclasses.replace(task.episode, high_level_period_s=0.1)

    assert dataclasses.replace(task, success=success, episode=episode).required_hold_steps() == 10


def test_layout_places_goal_at_sampled_distance_and_direction() -> None:
    reset = PushResetConfig(
        box_xy_jitter_m=0.0,
        goal_distance_m=ValueRange(low=2.0, high=2.0),
        goal_direction_deg=ValueRange(low=90.0, high=90.0),
    )

    layout = sample_push_layout((1.0, 0.5), reset, num_envs=3, generator=torch.Generator().manual_seed(0))

    assert torch.allclose(layout.box_xy_m, torch.tensor([[1.0, 0.5]] * 3))
    assert torch.allclose(layout.goal_xy_m, torch.tensor([[1.0, 2.5]] * 3), atol=1e-6)


def test_layout_jitter_and_ranges_stay_within_bounds_and_are_reproducible() -> None:
    reset = PushResetConfig(
        box_xy_jitter_m=0.2,
        goal_distance_m=ValueRange(low=1.0, high=1.5),
        goal_direction_deg=ValueRange(low=-30.0, high=30.0),
    )

    first = sample_push_layout((0.0, 0.0), reset, num_envs=256, generator=torch.Generator().manual_seed(7))
    second = sample_push_layout((0.0, 0.0), reset, num_envs=256, generator=torch.Generator().manual_seed(7))

    offset = first.goal_xy_m - first.box_xy_m
    distance = torch.linalg.vector_norm(offset, dim=1)
    direction_deg = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0]))
    assert bool((first.box_xy_m.abs() <= 0.2).all())
    assert bool(((distance >= 1.0 - 1e-6) & (distance <= 1.5 + 1e-6)).all())
    assert bool((direction_deg.abs() <= 30.0 + 1e-4).all())
    assert torch.equal(first.goal_xy_m, second.goal_xy_m)


def test_reset_config_rejects_non_positive_goal_distance() -> None:
    with pytest.raises(InvalidPushConfigError, match="reset.goal_distance_m.low"):
        PushResetConfig(
            box_xy_jitter_m=0.0,
            goal_distance_m=ValueRange(low=0.0, high=1.0),
            goal_direction_deg=ValueRange(low=0.0, high=math.pi),
        )
