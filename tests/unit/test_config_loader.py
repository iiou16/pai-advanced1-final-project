import dataclasses
from pathlib import Path

import pytest
import yaml

from cooperative_go2.entities.gait_config import InvalidGaitConfigError
from cooperative_go2.entities.gait_evaluation import InvalidEvaluationScenarioError
from cooperative_go2.entities.go2 import GO2_JOINT_NAMES
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_gait_evaluation_suite,
    load_gait_settings,
    parse_gait_settings,
)
from cooperative_go2.interface_adapters.rsl_rl_config import InvalidRslRlConfigError


def test_repository_gait_config_is_valid(gait_config_path: Path) -> None:
    load_gait_settings(gait_config_path)


def test_joint_angles_are_read_by_name_into_fixed_joint_order(gait_config_data: dict) -> None:
    angles = gait_config_data["robot"]["default_joint_angles_rad"]
    for index, name in enumerate(reversed(GO2_JOINT_NAMES)):
        angles[name] = float(index)
    shuffled = dict(reversed(list(angles.items())))
    gait_config_data["robot"]["default_joint_angles_rad"] = shuffled

    settings = parse_gait_settings(gait_config_data)

    in_order = settings.gait.robot.default_joint_angles_rad.in_order()
    assert in_order == tuple(float(index) for index in reversed(range(len(GO2_JOINT_NAMES))))


def test_loaded_config_cannot_be_modified(gait_config_data: dict) -> None:
    settings = parse_gait_settings(gait_config_data)

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.gait.robot.default_joint_angles_rad.FR_hip_joint = 1.0  # type: ignore[misc]


def test_integer_written_for_float_field_is_accepted_as_float(gait_config_data: dict) -> None:
    gait_config_data["env"]["episode_length_s"] = 20

    settings = parse_gait_settings(gait_config_data)

    assert isinstance(settings.gait.env.episode_length_s, float)


@pytest.mark.parametrize(
    ("section", "key"),
    [("reward", "tracking_sigma"), ("rsl_rl", "save_interval"), ("robot", "default_joint_angles_rad")],
    ids=["gait-key", "rsl-rl-key", "joint-angles"],
)
def test_missing_key_is_rejected(gait_config_data: dict, section: str, key: str) -> None:
    del gait_config_data[section][key]

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


def test_missing_rsl_rl_section_is_rejected(gait_config_data: dict) -> None:
    del gait_config_data["rsl_rl"]

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


@pytest.mark.parametrize(
    ("section", "key"), [("env", "num_env"), ("rsl_rl", "batch_size")], ids=["gait-typo", "rsl-rl-typo"]
)
def test_unknown_key_is_rejected(gait_config_data: dict, section: str, key: str) -> None:
    gait_config_data[section][key] = 16

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


def test_missing_joint_in_default_angles_is_rejected(gait_config_data: dict) -> None:
    del gait_config_data["robot"]["default_joint_angles_rad"][GO2_JOINT_NAMES[0]]

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("env", "num_envs", 16.5),
        ("env", "num_envs", True),
        ("env", "simulate_action_latency", "yes"),
        ("env", "control_dt_s", "0.02"),
        ("rsl_rl", "actor_hidden_dims", 512),
    ],
    ids=["float-for-int", "bool-for-int", "str-for-bool", "str-for-float", "scalar-for-list"],
)
def test_wrong_type_is_rejected(gait_config_data: dict, section: str, key: str, value: object) -> None:
    gait_config_data[section][key] = value

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("env", "num_envs", 0), ("env", "control_dt_s", -0.02), ("robot", "kp", 0.0)],
    ids=["zero-envs", "negative-dt", "zero-kp"],
)
def test_out_of_range_gait_value_is_rejected(gait_config_data: dict, section: str, key: str, value: object) -> None:
    gait_config_data[section][key] = value

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_gait_settings(gait_config_data)
    assert isinstance(exc_info.value.__cause__, InvalidGaitConfigError)


@pytest.mark.parametrize(
    ("key", "value"),
    [("gamma", 1.5), ("schedule", "cosine"), ("activation", "unknown"), ("critic_hidden_dims", [256, 0])],
    ids=["gamma-above-one", "unknown-schedule", "unknown-activation", "zero-width-layer"],
)
def test_invalid_rsl_rl_value_is_rejected(gait_config_data: dict, key: str, value: object) -> None:
    gait_config_data["rsl_rl"][key] = value

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_gait_settings(gait_config_data)
    assert isinstance(exc_info.value.__cause__, InvalidRslRlConfigError)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("command", "yaw_rate_radps", {"low": 1.0, "high": -1.0}),
        ("command", "resampling_time_s", 4.01),
        ("robot", "base_init_pos_m", [0.0, 0.42]),
        ("command", "zero_lin_vel_probability", 1.5),
        ("command", "zero_yaw_rate_probability", -0.1),
        ("env", "reset_joint_pos_noise_rad", -0.1),
    ],
    ids=[
        "range-low-above-high", "resampling-not-multiple-of-dt", "position-with-two-elements",
        "probability-above-one", "negative-probability", "negative-reset-noise",
    ],  # fmt: skip
)
def test_inconsistent_gait_value_is_rejected(gait_config_data: dict, section: str, key: str, value: object) -> None:
    gait_config_data[section][key] = value

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_gait_settings(gait_config_data)
    assert isinstance(exc_info.value.__cause__, InvalidGaitConfigError)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("env", "control_dt_s", float("nan")),
        ("reward", "tracking_sigma", float("inf")),
        ("rsl_rl", "learning_rate", float("nan")),
        ("command", "lin_vel_x_mps", {"low": float("nan"), "high": 1.0}),
    ],
    ids=["nan-dt", "inf-sigma", "nan-learning-rate", "nan-range-bound"],
)
def test_non_finite_number_is_rejected(gait_config_data: dict, section: str, key: str, value: object) -> None:
    gait_config_data[section][key] = value

    with pytest.raises(ConfigValidationError):
        parse_gait_settings(gait_config_data)


def test_yaml_nan_literal_is_rejected(tmp_path: Path, gait_config_path: Path) -> None:
    text = gait_config_path.read_text(encoding="utf-8").replace("action_scale: 0.25", "action_scale: .nan")
    config_path = tmp_path / "nan.yaml"
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_gait_settings(config_path)


def test_non_mapping_root_is_rejected() -> None:
    with pytest.raises(ConfigValidationError):
        parse_gait_settings(["env"])


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("env: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigValidationError) as exc_info:
        load_gait_settings(config_path)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gait_settings(tmp_path / "absent.yaml")


def test_repository_evaluation_scenarios_are_valid(gait_eval_config_path: Path) -> None:
    suite = load_gait_evaluation_suite(gait_eval_config_path)

    names = [scenario.name for scenario in suite.scenarios]
    assert {"stop", "forward", "turn_in_place_left", "turn_in_place_right", "direction_change"} <= set(names)


def test_evaluation_scenarios_within_training_command_range(
    gait_eval_config_path: Path, gait_config_path: Path
) -> None:
    command_config = load_gait_settings(gait_config_path).gait.command
    ranges = (command_config.lin_vel_x_mps, command_config.lin_vel_y_mps, command_config.yaw_rate_radps)

    suite = load_gait_evaluation_suite(gait_eval_config_path)

    for scenario in suite.scenarios:
        for segment in scenario.segments:
            for value, value_range in zip(segment.command.as_tuple(), ranges, strict=True):
                assert value_range.low <= value <= value_range.high, scenario.name


def _write_scenarios(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "scenarios.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_evaluation_scenario_with_duplicate_names_is_rejected(tmp_path: Path) -> None:
    segment = {"command": {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_rate_radps": 0.0}, "duration_s": 2.0}
    scenario = {"name": "stop", "segments": [segment]}
    path = _write_scenarios(tmp_path, {"settle_time_s": 1.0, "scenarios": [scenario, scenario]})

    with pytest.raises(ConfigValidationError) as exc_info:
        load_gait_evaluation_suite(path)
    assert isinstance(exc_info.value.__cause__, InvalidEvaluationScenarioError)


def test_evaluation_scenario_with_missing_command_key_is_rejected(tmp_path: Path) -> None:
    segment = {"command": {"vx_mps": 0.0, "yaw_rate_radps": 0.0}, "duration_s": 2.0}
    path = _write_scenarios(tmp_path, {"settle_time_s": 1.0, "scenarios": [{"name": "stop", "segments": [segment]}]})

    with pytest.raises(ConfigValidationError):
        load_gait_evaluation_suite(path)
