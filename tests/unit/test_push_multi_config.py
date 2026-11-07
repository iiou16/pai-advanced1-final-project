from pathlib import Path

import pytest
import yaml
from conftest import REPO_ROOT

from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_team_marl_run_settings,
    load_team_run_settings,
)

PUSH_MULTI = REPO_ROOT / "configs" / "push_multi.yaml"
TEAM_RULE_BASED = REPO_ROOT / "configs" / "team_rule_based.yaml"


def test_learning_and_rule_based_runs_share_the_comparison_conditions() -> None:
    # E4〜E7 を同じ条件で比べるため、シーン・開始位置の散らし方・周期・転倒・時間切れ・成功の維持時間をそろえる
    marl = load_team_marl_run_settings(PUSH_MULTI)
    rule_based = load_team_run_settings(TEAM_RULE_BASED)
    task = marl.settings.task
    team = rule_based.settings.team

    assert marl.settings.scene_config == rule_based.settings.scene_config
    assert marl.settings.gait_policy == rule_based.settings.gait_policy
    assert task.reset == team.reset
    assert task.episode.high_level_period_s == team.high_level_period_s
    assert task.episode.fall_roll_deg == team.fall_tilt_deg
    assert task.episode.fall_pitch_deg == team.fall_tilt_deg
    assert task.episode.episode_length_s == team.coordination.timeouts.task_s
    assert task.success_hold_time_s == team.coordination.verify_hold_s
    assert task.commands.vx_mps.high == team.push_control.push_speed_mps
    # 近づく報酬の目標の押す位置は、規則ベースの協調の押す位置と同じ
    assert task.reward.push_standoff_m == team.push_control.standoff_m
    assert tuple(sign * task.reward.push_lateral_offset_m for sign in task.agent_side_signs) == (
        team.push_lateral_offsets_m
    )
    # 規則ベースの協調の判断が相手の状態を受け取る遅れ（1 周期前の判断の結果）と同じ
    assert task.communication.delay_s == team.high_level_period_s


def test_agent_side_signs_must_match_the_scene_robots(tmp_path: Path) -> None:
    data = yaml.safe_load(PUSH_MULTI.read_text(encoding="utf-8"))
    data["task"]["agent_side_signs"] = [1.0]
    path = tmp_path / "push_multi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_team_marl_run_settings(path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("communication", "delay_s", 0.05), ("communication", "drop_probability", 1.0)],
    ids=["delay-shorter-than-a-period", "drop-everything"],
)
def test_invalid_communication_is_rejected(tmp_path: Path, section: str, key: str, value: float) -> None:
    data = yaml.safe_load(PUSH_MULTI.read_text(encoding="utf-8"))
    data["task"][section][key] = value
    path = tmp_path / "push_multi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_team_marl_run_settings(path)


def test_same_side_for_both_robots_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(PUSH_MULTI.read_text(encoding="utf-8"))
    data["task"]["agent_side_signs"] = [1.0, 1.0]
    path = tmp_path / "push_multi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_team_marl_run_settings(path)
