import pytest

from cooperative_go2.entities.push_config import (
    InvalidPushConfigError,
    PushArenaConfig,
    PushObservationScales,
    PushRewardConfig,
    PushRewardScales,
    PushSuccessConfig,
)


def _reward_scales() -> PushRewardScales:
    return PushRewardScales(
        box_progress=1.0,
        success=1.0,
        intended_face_contact=0.1,
        box_rotation=-0.1,
        robot_fall=-1.0,
        unsafe_collision=-1.0,
        control_effort=-0.01,
    )


def test_required_hold_steps_counts_exact_multiple_without_rounding_up() -> None:
    success = PushSuccessConfig(goal_radius_m=0.3, hold_time_s=0.3)

    # 0.3 / 0.1 は浮動小数点では 2.9999999999999996 になるが、3 ステップとして数える
    assert success.required_hold_steps(control_dt_s=0.1) == 3


def test_required_hold_steps_rounds_partial_step_up() -> None:
    success = PushSuccessConfig(goal_radius_m=0.3, hold_time_s=0.25)

    assert success.required_hold_steps(control_dt_s=0.1) == 3


def test_required_hold_steps_rejects_non_positive_control_period() -> None:
    success = PushSuccessConfig(goal_radius_m=0.3, hold_time_s=1.0)

    with pytest.raises(InvalidPushConfigError, match="control_dt_s"):
        success.required_hold_steps(control_dt_s=0.0)


@pytest.mark.parametrize(
    ("goal_radius_m", "hold_time_s", "key"),
    [(0.0, 1.0, "success.goal_radius_m"), (0.3, -1.0, "success.hold_time_s")],
)
def test_success_config_rejects_non_positive_values(goal_radius_m: float, hold_time_s: float, key: str) -> None:
    with pytest.raises(InvalidPushConfigError, match=key):
        PushSuccessConfig(goal_radius_m=goal_radius_m, hold_time_s=hold_time_s)


def test_arena_config_rejects_non_positive_extent() -> None:
    with pytest.raises(InvalidPushConfigError, match="arena.half_extent_y_m"):
        PushArenaConfig(half_extent_x_m=5.0, half_extent_y_m=0.0)


def test_observation_scales_reject_non_positive_value() -> None:
    with pytest.raises(InvalidPushConfigError, match="observation.ang_vel"):
        PushObservationScales(position=1.0, lin_vel=2.0, ang_vel=0.0)


@pytest.mark.parametrize("min_cos", [-1.0, 1.0])
def test_reward_config_accepts_cosine_bounds(min_cos: float) -> None:
    config = PushRewardConfig(intended_face_min_cos=min_cos, scales=_reward_scales())

    assert config.intended_face_min_cos == min_cos


@pytest.mark.parametrize("min_cos", [-1.01, 1.01])
def test_reward_config_rejects_cosine_outside_range(min_cos: float) -> None:
    with pytest.raises(InvalidPushConfigError, match="reward.intended_face_min_cos"):
        PushRewardConfig(intended_face_min_cos=min_cos, scales=_reward_scales())
