"""箱押しの成功と終了の判定（学習環境と ROS 2 のシミュレーションノードで共有する）の検証。"""

from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.push_episode_judge import PushJudgeInputs, judge_push_step
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.interface_adapters.config_loader import load_push_training_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = (2.0, 0.0)


@pytest.fixture
def task() -> PushTaskConfig:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single_wide.yaml").task


def _inputs(
    *, box_xy: tuple[float, float] = (1.0, 0.0), robot_xy: tuple[float, float] = (0.0, 0.0), episode_steps: int = 1,
    hold_steps: int = 0, roll_deg: float = 0.0, diverged: bool = False,
) -> PushJudgeInputs:  # fmt: skip
    return PushJudgeInputs(
        robot_xy_m=torch.tensor([robot_xy]),
        box_xy_m=torch.tensor([box_xy]),
        goal_xy_m=torch.tensor([GOAL]),
        base_euler_deg=torch.tensor([[roll_deg, 0.0, 0.0]]),
        episode_steps=torch.tensor([episode_steps]),
        hold_steps=torch.tensor([hold_steps]),
        diverged=torch.tensor([diverged]),
    )


def _reason(task: PushTaskConfig, inputs: PushJudgeInputs) -> PushTerminationReason:
    return PushTerminationReason(int(judge_push_step(inputs, task).reasons[0]))


def test_box_away_from_goal_does_not_end_episode(task: PushTaskConfig) -> None:
    judgement = judge_push_step(_inputs(), task)

    assert PushTerminationReason(int(judgement.reasons[0])) is PushTerminationReason.NONE
    assert int(judgement.hold_steps[0]) == 0


def test_success_after_holding_goal_for_required_steps(task: PushTaskConfig) -> None:
    required = task.required_hold_steps()

    almost = judge_push_step(_inputs(box_xy=GOAL, hold_steps=required - 2), task)
    held = judge_push_step(_inputs(box_xy=GOAL, hold_steps=required - 1), task)

    assert PushTerminationReason(int(almost.reasons[0])) is PushTerminationReason.NONE
    assert PushTerminationReason(int(held.reasons[0])) is PushTerminationReason.SUCCESS


def test_hold_resets_when_box_leaves_goal(task: PushTaskConfig) -> None:
    judgement = judge_push_step(_inputs(box_xy=(1.0, 0.0), hold_steps=5), task)

    assert int(judgement.hold_steps[0]) == 0


def test_time_out_at_max_episode_length(task: PushTaskConfig) -> None:
    max_steps = task.episode.max_episode_length_steps()

    assert _reason(task, _inputs(episode_steps=max_steps - 1)) is PushTerminationReason.NONE
    assert _reason(task, _inputs(episode_steps=max_steps)) is PushTerminationReason.TIME_OUT


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"roll_deg": 45.0}, PushTerminationReason.FALL),
        ({"box_xy": (100.0, 0.0)}, PushTerminationReason.OUT_OF_BOUNDS),
        ({"robot_xy": (0.0, -100.0)}, PushTerminationReason.OUT_OF_BOUNDS),
        ({"diverged": True}, PushTerminationReason.SIM_DIVERGED),
    ],
    ids=["fall", "box_out_of_arena", "robot_out_of_arena", "diverged"],
)
def test_failure_reasons(task: PushTaskConfig, overrides: dict, expected: PushTerminationReason) -> None:
    assert _reason(task, _inputs(**overrides)) is expected


def test_divergence_takes_priority_over_success(task: PushTaskConfig) -> None:
    inputs = _inputs(box_xy=GOAL, hold_steps=task.required_hold_steps(), diverged=True)

    assert _reason(task, inputs) is PushTerminationReason.SIM_DIVERGED
