import dataclasses
from pathlib import Path

import pytest

from conftest import REPO_ROOT

from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.entities.push_robustness import (
    PushRobustnessCondition,
    PushRobustnessSuite,
    UnknownRobustnessConditionError,
)
from cooperative_go2.entities.push_task_config import PushResetConfig
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_push_robustness_suite,
    load_push_scene_settings,
    load_push_training_settings,
)

ROBUSTNESS_CONFIG = REPO_ROOT / "configs" / "push_robustness.yaml"
EVALUATION_CONFIG = REPO_ROOT / "configs" / "push_single_wide.yaml"
NOMINAL = "nominal"


def _condition(name: str, command_delay_gait_steps: int = 0) -> PushRobustnessCondition:
    reset = PushResetConfig(
        box_xy_jitter_m=0.1, goal_distance_m=ValueRange(1.0, 1.0), goal_direction_deg=ValueRange(-60.0, 60.0)
    )
    return PushRobustnessCondition(
        name=name, box_mass_kg=2.0, box_friction=1.0, reset=reset, command_delay_gait_steps=command_delay_gait_steps
    )


def _changed_factors(condition: PushRobustnessCondition, nominal: PushRobustnessCondition) -> list[str]:
    factors = [field.name for field in dataclasses.fields(PushRobustnessCondition) if field.name != "name"]
    return [factor for factor in factors if getattr(condition, factor) != getattr(nominal, factor)]


def test_nominal_condition_is_the_training_condition_of_the_evaluated_policy() -> None:
    nominal = load_push_robustness_suite(ROBUSTNESS_CONFIG).get_condition(NOMINAL)
    settings = load_push_training_settings(EVALUATION_CONFIG)
    box = load_push_scene_settings(REPO_ROOT / settings.scene_config).box

    assert nominal.reset == settings.task.reset
    assert (nominal.box_mass_kg, nominal.box_friction) == (box.mass_kg, box.friction)
    assert nominal.command_delay_gait_steps == 0


def test_every_other_condition_changes_exactly_one_factor_from_nominal() -> None:
    suite = load_push_robustness_suite(ROBUSTNESS_CONFIG)
    nominal = suite.get_condition(NOMINAL)

    changed = {condition.name: _changed_factors(condition, nominal) for condition in suite.conditions}

    assert {name: factors for name, factors in changed.items() if name != NOMINAL and len(factors) != 1} == {}


def test_unknown_condition_name_is_rejected() -> None:
    suite = PushRobustnessSuite(conditions=(_condition(NOMINAL),))

    with pytest.raises(UnknownRobustnessConditionError, match="absent"):
        suite.get_condition("absent")


def test_duplicated_condition_names_are_rejected() -> None:
    with pytest.raises(InvalidPushConfigError, match="重複"):
        PushRobustnessSuite(conditions=(_condition(NOMINAL), _condition(NOMINAL)))


@pytest.mark.parametrize("command_delay_gait_steps", [-1, 1.5, True])
def test_command_delay_that_is_not_a_non_negative_integer_is_rejected(command_delay_gait_steps: object) -> None:
    with pytest.raises(InvalidPushConfigError, match="command_delay_gait_steps"):
        _condition("delay", command_delay_gait_steps=command_delay_gait_steps)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["box_mass_kg", "box_friction"])
def test_non_positive_box_mass_or_friction_is_rejected(field: str) -> None:
    with pytest.raises(InvalidPushConfigError, match=field):
        dataclasses.replace(_condition(NOMINAL), **{field: 0.0})


def test_condition_with_missing_key_is_rejected_on_load(tmp_path: Path) -> None:
    original = ROBUSTNESS_CONFIG.read_text(encoding="utf-8")
    text = original.replace("    command_delay_gait_steps: 0\n", "", 1)
    assert text != original
    config = tmp_path / "robustness.yaml"
    config.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="command_delay_gait_steps"):
        load_push_robustness_suite(config)


def test_empty_condition_name_is_rejected() -> None:
    with pytest.raises(InvalidPushConfigError, match="name"):
        dataclasses.replace(_condition(NOMINAL), name="")
