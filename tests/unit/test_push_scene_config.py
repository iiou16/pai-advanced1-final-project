from pathlib import Path

import pytest
import yaml

from cooperative_go2.entities.push_scene_config import BoxConfig, InvalidPushSceneConfigError
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_push_scene_settings,
    parse_push_scene_settings,
)

SCENE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "s1_push.yaml"
VALID_BOX = {"size_m": (0.4, 0.4, 0.3), "mass_kg": 2.0, "friction": 1.0, "initial_xy_m": (1.0, 0.0)}


def _scene_data() -> dict:
    return yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))


def test_repository_scene_config_is_valid() -> None:
    settings = load_push_scene_settings(SCENE_CONFIG_PATH)

    assert len(settings.box.size_m) == 3


def test_box_density_gives_configured_mass() -> None:
    box = BoxConfig(size_m=(0.5, 0.4, 0.2), mass_kg=2.0, friction=1.0, initial_xy_m=(1.0, 0.0))

    assert box.density_kg_per_m3() * box.volume_m3() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"size_m": (0.4, 0.4)}, "box.size_m"),
        ({"size_m": (0.4, 0.0, 0.3)}, "box.size_m.y"),
        ({"mass_kg": -1.0}, "box.mass_kg"),
        ({"friction": 0.0}, "box.friction"),
        ({"initial_xy_m": (1.0, 0.0, 0.0)}, "box.initial_xy_m"),
    ],
)
def test_box_config_rejects_invalid_value(overrides: dict, key: str) -> None:
    with pytest.raises(InvalidPushSceneConfigError, match=key):
        BoxConfig(**(VALID_BOX | overrides))


@pytest.mark.parametrize("friction", [0.005, 5.5])
def test_box_friction_outside_genesis_range_is_rejected_at_load(friction: float) -> None:
    data = _scene_data()
    data["box"]["friction"] = friction

    # 正の値ならドメインとしては受け付けるが、Genesis が受け付けない値は読み込み時に弾く
    with pytest.raises(ConfigValidationError, match="box.friction"):
        parse_push_scene_settings(data)


def test_invalid_box_value_is_reported_with_key_path() -> None:
    data = _scene_data()
    data["box"]["mass_kg"] = 0.0

    with pytest.raises(ConfigValidationError, match="box: box.mass_kg"):
        parse_push_scene_settings(data)


def test_non_positive_collision_pairs_is_rejected() -> None:
    data = _scene_data()
    data["genesis"]["max_collision_pairs"] = 0

    with pytest.raises(ConfigValidationError, match="genesis.max_collision_pairs"):
        parse_push_scene_settings(data)


def test_unknown_key_is_rejected() -> None:
    data = _scene_data()
    data["box"]["color"] = "red"

    with pytest.raises(ConfigValidationError, match="未知"):
        parse_push_scene_settings(data)
