import pytest

from cooperative_go2.interface_adapters.config_loader import parse_gait_settings


@pytest.mark.parametrize(
    ("episode_length_s", "control_dt_s", "expected_steps"),
    [(20.0, 0.02, 1000), (1.0, 0.3, 4), (0.9, 0.3, 3)],
    ids=["divisible-with-rounding-error", "rounds-up", "ratio-slightly-above-integer"],
)
def test_episode_length_in_steps_is_ceiling_tolerant_to_rounding(
    gait_config_data: dict, episode_length_s: float, control_dt_s: float, expected_steps: int
) -> None:
    gait_config_data["env"]["episode_length_s"] = episode_length_s
    gait_config_data["env"]["control_dt_s"] = control_dt_s
    gait_config_data["command"]["resampling_time_s"] = control_dt_s

    config = parse_gait_settings(gait_config_data).gait

    assert config.max_episode_length_steps() == expected_steps


def test_command_resampling_interval_is_in_control_steps(gait_config_data: dict) -> None:
    gait_config_data["env"]["control_dt_s"] = 0.02
    gait_config_data["command"]["resampling_time_s"] = 4.0

    config = parse_gait_settings(gait_config_data).gait

    assert config.command_resampling_interval_steps() == 200
