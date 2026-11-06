"""規則ベースで箱を押す作業の設定（不変条件・読み込み・押す位置と箱の整合）と、開始位置の引き方の検証。"""

import dataclasses
import math
from pathlib import Path

import pytest
import torch
import yaml

from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.entities.team_task_config import (
    InvalidTeamTaskConfigError,
    TaskManagerConfig,
    TeamResetConfig,
    TeamRuleBasedConfig,
    sample_team_start_positions,
)
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_ros2_team_settings,
    load_team_run_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CONFIG = Path("configs/team_rule_based.yaml")


@pytest.fixture
def repo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """設定ファイルが相対パスでシーンの設定を指すので、リポジトリのルートで読む。"""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture
def team(repo_root: Path) -> TeamRuleBasedConfig:
    return load_team_run_settings(TEAM_CONFIG).settings.team


@pytest.mark.parametrize(
    ("ros2_config", "num_robots"),
    [("configs/ros2_team.yaml", 2), ("configs/ros2_team_single.yaml", 1)],
)
def test_repository_team_configs_load(repo_root: Path, ros2_config: str, num_robots: int) -> None:
    team_config = Path(load_ros2_team_settings(Path(ros2_config)).team_config)

    run = load_team_run_settings(team_config)

    assert len(run.scene.scene.robots) == num_robots
    assert len(run.settings.team.push_lateral_offsets_m) == num_robots


def write_team_config(tmp_path: Path, repo_root: Path, change: dict[str, object]) -> Path:
    data = yaml.safe_load((repo_root / TEAM_CONFIG).read_text(encoding="utf-8"))
    for dotted_key, value in change.items():
        *parents, last = dotted_key.split(".")
        target = data
        for key in parents:
            target = target[key]
        target[last] = value
    path = tmp_path / "team.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "change",
    [
        # 2 台のシーンに押す側が 1 台分しかない
        {"team.push_lateral_offsets_m": [0.0]},
        # 押す位置が箱の中（箱の奥行きの半分 0.2 m 以下）
        {"team.push_control.standoff_m": 0.2},
        # 押す線の横のずれとずらしの上限の和（0.4 + 0.2）が箱の幅の半分 0.5 m を超える
        {"team.push_lateral_offsets_m": [0.4, -0.25]},
        # 型が違う
        {"team.task_manager.timeout_s": "long"},
    ],
    ids=["offsets-count", "standoff-inside-box", "push-line-off-box", "wrong-type"],
)
def test_loader_rejects_inconsistent_team_config(
    tmp_path: Path, repo_root: Path, change: dict[str, object]
) -> None:
    path = write_team_config(tmp_path, repo_root, change)

    with pytest.raises(ConfigValidationError):
        load_team_run_settings(path)


def test_loader_reports_missing_scene_config(tmp_path: Path, repo_root: Path) -> None:
    path = write_team_config(tmp_path, repo_root, {"scene_config": "configs/scenes/missing.yaml"})

    with pytest.raises(FileNotFoundError):
        load_team_run_settings(path)


@pytest.mark.parametrize("timeout_s", [60.0, 59.0], ids=["equal-to-task", "shorter-than-task"])
def test_task_manager_timeout_must_exceed_robot_task_timeout(team: TeamRuleBasedConfig, timeout_s: float) -> None:
    with pytest.raises(InvalidTeamTaskConfigError):
        dataclasses.replace(team, task_manager=TaskManagerConfig(timeout_s=timeout_s))


def test_task_manager_timeout_longer_than_robot_task_timeout_is_accepted(team: TeamRuleBasedConfig) -> None:
    config = dataclasses.replace(team, task_manager=TaskManagerConfig(timeout_s=60.5))

    assert config.task_manager.timeout_s == 60.5


@pytest.mark.parametrize("field", ["partner_stale_after_s", "push_start_grace_s"])
def test_readiness_times_must_exceed_decision_period(team: TeamRuleBasedConfig, field: str) -> None:
    readiness = dataclasses.replace(team.team_readiness, **{field: team.high_level_period_s})

    with pytest.raises(InvalidTeamTaskConfigError):
        dataclasses.replace(team, team_readiness=readiness)


@pytest.mark.parametrize(
    "change",
    [
        {"push_lateral_offsets_m": ()},
        {"push_lateral_offsets_m": (0.25, math.nan)},
        {"high_level_period_s": 0.0},
        {"fall_tilt_deg": 0.0},
    ],
    ids=["no-offsets", "nan-offset", "zero-period", "zero-fall-tilt"],
)
def test_team_config_rejects_invalid_values(team: TeamRuleBasedConfig, change: dict[str, object]) -> None:
    with pytest.raises(InvalidTeamTaskConfigError):
        dataclasses.replace(team, **change)


def test_gait_steps_per_decision(team: TeamRuleBasedConfig) -> None:
    assert team.gait_steps_per_high_level_step(0.02) == 5


def test_gait_steps_reject_non_integer_ratio(team: TeamRuleBasedConfig) -> None:
    with pytest.raises(InvalidTeamTaskConfigError):
        team.gait_steps_per_high_level_step(0.03)


RESET = TeamResetConfig(start_back_m=ValueRange(low=0.0, high=0.8), start_lateral_jitter_m=0.05)
INITIAL_XY = ((0.7, 0.25), (0.7, -0.25))


def sample(seed: int, num_envs: int = 64) -> tuple[torch.Tensor, ...]:
    return sample_team_start_positions(INITIAL_XY, RESET, num_envs, torch.Generator().manual_seed(seed))


def test_start_positions_stay_behind_initial_positions_within_jitter() -> None:
    positions = sample(seed=1)

    for (x_m, y_m), robot in zip(INITIAL_XY, positions, strict=True):
        assert robot.shape == (64, 2)
        assert bool(((robot[:, 0] <= x_m) & (robot[:, 0] >= x_m - 0.8)).all())
        assert bool(((robot[:, 1] - y_m).abs() <= 0.05).all())


def test_start_positions_are_reproducible_with_same_seed() -> None:
    first, second = sample(seed=7), sample(seed=7)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_start_positions_are_drawn_independently_per_robot() -> None:
    robot_a, robot_b = sample(seed=3)

    # 後ろへのずれが 2 台で同じなら、押す位置に着く時刻がずれない
    assert not torch.allclose(0.7 - robot_a[:, 0], 0.7 - robot_b[:, 0])
