import dataclasses

import pytest

from conftest import REPO_ROOT

from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.interface_adapters.config_loader import (
    PushSceneSettings,
    PushTrainingSettings,
    load_push_robustness_suite,
    load_push_scene_settings,
    load_push_training_settings,
)
from cooperative_go2.interface_adapters.push_robustness_settings import (
    IneffectiveBoxFrictionError,
    apply_robustness_condition,
)

FLOOR_FRICTION = 1.0


@pytest.fixture
def settings() -> PushTrainingSettings:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single_wide.yaml")


@pytest.fixture
def scene(settings: PushTrainingSettings) -> PushSceneSettings:
    return load_push_scene_settings(REPO_ROOT / settings.scene_config)


def _condition(name: str):
    return load_push_robustness_suite(REPO_ROOT / "configs" / "push_robustness.yaml").get_condition(name)


def test_box_mass_and_friction_are_replaced_by_condition(
    settings: PushTrainingSettings, scene: PushSceneSettings
) -> None:
    condition = dataclasses.replace(_condition("nominal"), box_mass_kg=3.5, box_friction=1.2)

    applied = apply_robustness_condition(settings, scene, condition, FLOOR_FRICTION)

    assert (applied.scene.box.mass_kg, applied.scene.box.friction) == (3.5, 1.2)
    assert applied.scene.box.size_m == scene.box.size_m


def test_placement_is_replaced_by_condition(settings: PushTrainingSettings, scene: PushSceneSettings) -> None:
    condition = _condition("goal_direction_90deg")

    applied = apply_robustness_condition(settings, scene, condition, FLOOR_FRICTION)

    assert applied.settings.task.reset.goal_direction_deg == ValueRange(-90.0, 90.0)
    assert applied.settings.task.commands == settings.task.commands


def test_box_friction_equal_to_floor_is_accepted(settings: PushTrainingSettings, scene: PushSceneSettings) -> None:
    condition = dataclasses.replace(_condition("nominal"), box_friction=FLOOR_FRICTION)

    applied = apply_robustness_condition(settings, scene, condition, FLOOR_FRICTION)

    assert applied.scene.box.friction == FLOOR_FRICTION


def test_box_friction_below_floor_is_rejected(settings: PushTrainingSettings, scene: PushSceneSettings) -> None:
    condition = dataclasses.replace(_condition("nominal"), box_friction=0.9)

    with pytest.raises(IneffectiveBoxFrictionError, match="nominal"):
        apply_robustness_condition(settings, scene, condition, FLOOR_FRICTION)
