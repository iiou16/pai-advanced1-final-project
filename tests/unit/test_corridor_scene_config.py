import dataclasses
from pathlib import Path

import pytest
import yaml

from cooperative_go2.entities.corridor import CorridorConfig
from cooperative_go2.entities.corridor_scene_config import (
    CorridorSceneConfig,
    InvalidCorridorSceneConfigError,
    RetreatAreaConfig,
    RobotPlacement,
    WallConfig,
)
from cooperative_go2.entities.push_scene_config import BoxConfig
from cooperative_go2.interface_adapters.config_loader import ConfigValidationError, load_corridor_scene_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_CONFIG_DIR = REPO_ROOT / "configs" / "scenes"


@pytest.fixture
def valid_scene() -> CorridorSceneConfig:
    """通路 x ∈ [1, 2]・幅 1.4・通行領域 x ∈ [0.3, 2.7] に、通路をふさぐ箱と 1 台の Go2 を置いたシーン。"""
    return CorridorSceneConfig(
        box=BoxConfig(size_m=(0.4, 1.0, 0.3), mass_kg=2.0, friction=1.0, initial_xy_m=(1.4, 0.0)),
        corridor=CorridorConfig(entrance_x_m=1.0, exit_x_m=2.0, center_y_m=0.0, width_m=1.4, apron_m=0.7),
        walls=WallConfig(height_m=0.5, outer_half_extent_y_m=4.0),
        retreat_area=RetreatAreaConfig(center_xy_m=(3.8, 0.0), radius_m=0.4),
        robots=(RobotPlacement(name="go2_a", initial_xy_m=(0.7, 0.0)),),
    )


@pytest.mark.parametrize("filename", ["s2_corridor.yaml", "s3_team.yaml"])
def test_repository_scene_configs_load(filename: str) -> None:
    settings = load_corridor_scene_settings(SCENE_CONFIG_DIR / filename)

    assert settings.scene.robots


def test_team_scene_differs_from_single_scene_only_in_box_mass_and_robots() -> None:
    single = load_corridor_scene_settings(SCENE_CONFIG_DIR / "s2_corridor.yaml").scene
    team = load_corridor_scene_settings(SCENE_CONFIG_DIR / "s3_team.yaml").scene

    assert dataclasses.replace(team, box=single.box, robots=single.robots) == single
    assert dataclasses.replace(team.box, mass_kg=single.box.mass_kg) == single.box
    assert [robot.name for robot in team.robots] == ["go2_a", "go2_b"]


def test_loader_rejects_missing_genesis_section(tmp_path: Path) -> None:
    data = yaml.safe_load((SCENE_CONFIG_DIR / "s2_corridor.yaml").read_text(encoding="utf-8"))
    del data["genesis"]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_corridor_scene_settings(path)


def test_loader_converts_invariant_violation_to_config_error(tmp_path: Path) -> None:
    data = yaml.safe_load((SCENE_CONFIG_DIR / "s2_corridor.yaml").read_text(encoding="utf-8"))
    data["retreat_area"]["center_xy_m"] = [2.0, 0.0]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_corridor_scene_settings(path)


def test_rejects_box_that_does_not_block_corridor(valid_scene: CorridorSceneConfig) -> None:
    box_outside = dataclasses.replace(valid_scene.box, initial_xy_m=(5.0, 0.0))

    with pytest.raises(InvalidCorridorSceneConfigError):
        dataclasses.replace(valid_scene, box=box_outside)


@pytest.mark.parametrize(
    ("center", "radius"),
    # 通行領域の中、円の縁が通行領域の端に届く、円は離れているが縁に置いた箱（対角線の半分 ≈ 0.539）が回ると掛かる
    [((2.0, 0.0), 0.4), ((3.1, 0.0), 0.4), ((3.6, 0.0), 0.4)],
    ids=["inside-region", "circle-touching-region", "rotated-box-at-rim-reaches-region"],
)
def test_rejects_retreat_area_overlapping_passage_region(
    valid_scene: CorridorSceneConfig, center: tuple[float, float], radius: float
) -> None:
    with pytest.raises(InvalidCorridorSceneConfigError):
        dataclasses.replace(valid_scene, retreat_area=RetreatAreaConfig(center_xy_m=center, radius_m=radius))


def test_accepts_retreat_area_with_room_for_rotated_box(valid_scene: CorridorSceneConfig) -> None:
    # 円の縁 3.3 から箱の対角線の半分を引いても 2.761 > 2.7（通行領域の出口側の端）
    scene = dataclasses.replace(valid_scene, retreat_area=RetreatAreaConfig(center_xy_m=(3.7, 0.0), radius_m=0.4))

    assert scene.retreat_area.center_xy_m == (3.7, 0.0)


def test_rejects_duplicate_robot_names(valid_scene: CorridorSceneConfig) -> None:
    robots = (
        RobotPlacement(name="go2_a", initial_xy_m=(0.7, 0.25)),
        RobotPlacement(name="go2_a", initial_xy_m=(0.7, -0.25)),
    )

    with pytest.raises(InvalidCorridorSceneConfigError):
        dataclasses.replace(valid_scene, robots=robots)


def test_rejects_scene_without_robots(valid_scene: CorridorSceneConfig) -> None:
    with pytest.raises(InvalidCorridorSceneConfigError):
        dataclasses.replace(valid_scene, robots=())


def test_rejects_walls_not_reaching_beyond_corridor(valid_scene: CorridorSceneConfig) -> None:
    with pytest.raises(InvalidCorridorSceneConfigError):
        dataclasses.replace(valid_scene, walls=WallConfig(height_m=0.5, outer_half_extent_y_m=0.7))


def test_rejects_empty_robot_name() -> None:
    with pytest.raises(InvalidCorridorSceneConfigError):
        RobotPlacement(name="", initial_xy_m=(0.0, 0.0))
