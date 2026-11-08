"""協調のノードを学習した方策で動かすときの組み立てと、学習した方策の判断の段階への写し方の検証。"""

import math
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.frameworks import ros2_team_coord_node
from cooperative_go2.interface_adapters.config_loader import load_team_marl_run_settings
from cooperative_go2.interface_adapters.ros2_team_deciders import LEARNED_MODE, LearnedTeamDecider
from cooperative_go2.use_cases.team_policy_driver import TeamPolicyDriver
from cooperative_go2.use_cases.team_push_agent import RetreatArea

REPO_ROOT = Path(__file__).resolve().parents[2]
ROS2_TEAM_CONFIG = REPO_ROOT / "configs" / "ros2_team.yaml"
PUSH_MULTI = REPO_ROOT / "configs" / "push_multi.yaml"
SCENE = REPO_ROOT / "configs" / "scenes" / "s3_team.yaml"
PERIOD_NS = 100_000_000
START_NS = 2_000_000_000
RETREAT = RetreatArea(center_xy_m=(3.8, 0.0), radius_m=0.4)


def _policy_run_dir(tmp_path: Path, edit: dict[str, object]) -> Path:
    # 学習の出力先の写し（push_multi.yaml）を、task の値を書き換えて置く
    data = yaml.safe_load(PUSH_MULTI.read_text(encoding="utf-8"))
    for key_path, value in edit.items():
        *parents, leaf = key_path.split(".")
        target = data
        for parent in parents:
            target = target[parent]
        target[leaf] = value
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "push_multi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(SCENE, run_dir / "scene.yaml")
    return run_dir


def _run_main(run_dir: Path, tmp_path: Path) -> int:
    arguments = [
        "go2-ros2-team-coord", "--config", str(ROS2_TEAM_CONFIG), "--mode", LEARNED_MODE,
        "--policy-run-dir", str(run_dir), "--policy-iteration", "50", "--metrics-dir", str(tmp_path / "metrics"),
    ]  # fmt: skip
    with pytest.raises(SystemExit) as exc_info:
        ros2_team_coord_node.main(arguments)
    return exc_info.value.code


def test_policy_trained_in_another_scene_is_rejected_before_starting_the_node(tmp_path: Path) -> None:
    # 学習の出力先に残したシーンの中身が実行系のシーンと違えば、ファイルの場所が同じでも止める
    run_dir = _policy_run_dir(tmp_path, {})
    archived = yaml.safe_load((run_dir / "scene.yaml").read_text(encoding="utf-8"))
    archived["box"]["mass_kg"] = 4.0
    (run_dir / "scene.yaml").write_text(yaml.safe_dump(archived, allow_unicode=True), encoding="utf-8")

    assert _run_main(run_dir, tmp_path) == ros2_team_coord_node.EXIT_INVALID_INPUT


@pytest.mark.parametrize(
    "edit",
    [
        {"task.episode.high_level_period_s": 0.2, "task.communication.delay_s": 0.2},
        {"task.communication.delay_s": 0.2},
    ],
    ids=["different-decision-period", "two-period-delay"],
)
def test_policy_trained_with_other_timing_is_rejected_before_starting_the_node(
    tmp_path: Path, edit: dict[str, object]
) -> None:
    run_dir = _policy_run_dir(tmp_path, edit)

    assert _run_main(run_dir, tmp_path) == ros2_team_coord_node.EXIT_INVALID_INPUT


def test_learned_mode_without_a_policy_is_rejected(tmp_path: Path) -> None:
    arguments = ["go2-ros2-team-coord", "--config", str(ROS2_TEAM_CONFIG), "--mode", LEARNED_MODE,
                 "--metrics-dir", str(tmp_path)]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        ros2_team_coord_node.main(arguments)

    assert exc_info.value.code != 0


def _pose(x_m: float, y_m: float, roll_rad: float = 0.0) -> RigidBodyPose:
    quat = torch.tensor([[math.cos(roll_rad / 2), math.sin(roll_rad / 2), 0.0, 0.0]])
    return RigidBodyPose.from_world_velocities(
        torch.tensor([[x_m, y_m, 0.3]]), quat, torch.zeros(1, 3), torch.zeros(1, 3)
    )


@pytest.fixture
def decider() -> LearnedTeamDecider:
    task = load_team_marl_run_settings(PUSH_MULTI).settings.task
    learned = LearnedTeamDecider(TeamPolicyDriver(lambda observation: torch.zeros(1, 3), task, robot_index=0))
    learned.start(RETREAT, START_NS)
    return learned


def test_learned_decider_reports_approach_while_the_box_is_outside_the_retreat_area(
    decider: LearnedTeamDecider,
) -> None:
    decision = decider.decide(START_NS, _pose(0.7, 0.25), _pose(1.4, 0.0))

    assert decision.phase is TeamPhase.APPROACH
    assert decision.failure_reason is None


def test_learned_decider_reports_verify_then_done_while_the_box_stays_in_the_retreat_area(
    decider: LearnedTeamDecider,
) -> None:
    task = load_team_marl_run_settings(PUSH_MULTI).settings.task

    phases = [
        decider.decide(START_NS + step * PERIOD_NS, _pose(3.0, 0.25), _pose(3.8, 0.0)).phase
        for step in range(task.required_hold_steps())
    ]

    assert phases[:-1] == [TeamPhase.VERIFY] * (task.required_hold_steps() - 1)
    assert phases[-1] is TeamPhase.DONE


def test_learned_decider_reports_the_failure_reason(decider: LearnedTeamDecider) -> None:
    decision = decider.decide(START_NS, _pose(0.7, 0.25, roll_rad=math.radians(15.0)), _pose(1.4, 0.0))

    assert decision.phase is TeamPhase.FAILED
    assert decision.failure_reason == "FALL"
