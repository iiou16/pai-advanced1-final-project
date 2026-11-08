import dataclasses

import pytest
import torch

from cooperative_go2.entities.push_config import PushArenaConfig
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.team_marl_judge import TeamJudgeInputs, TeamJudgeRules, judge_team_step

NUM_ENVS = 2
NO_ENVS = torch.zeros(NUM_ENVS, dtype=torch.bool)
RULES = TeamJudgeRules(
    retreat_xy_m=(3.8, 0.0), retreat_radius_m=0.4, required_hold_steps=10, max_episode_steps=600,
    arena=PushArenaConfig(half_extent_x_m=5.0, half_extent_y_m=4.0),
)  # fmt: skip


@pytest.fixture
def inputs() -> TeamJudgeInputs:
    return TeamJudgeInputs(
        robots_xy_m=(torch.tensor([[0.8, 0.25]] * NUM_ENVS), torch.tensor([[0.8, -0.25]] * NUM_ENVS)),
        box_xy_m=torch.tensor([[1.4, 0.0]] * NUM_ENVS),
        fallen=(NO_ENVS, NO_ENVS),
        episode_steps=torch.ones(NUM_ENVS, dtype=torch.int64),
        hold_steps=torch.zeros(NUM_ENVS, dtype=torch.int64),
        diverged=NO_ENVS,
    )


def _reasons(inputs: TeamJudgeInputs) -> list[int]:
    return judge_team_step(inputs, RULES).reasons.tolist()


def test_episode_continues_without_termination(inputs: TeamJudgeInputs) -> None:
    assert _reasons(inputs) == [PushTerminationReason.NONE] * NUM_ENVS


def test_a_fall_of_either_robot_ends_the_shared_episode(inputs: TeamJudgeInputs) -> None:
    judged = dataclasses.replace(inputs, fallen=(NO_ENVS, torch.tensor([False, True])))

    assert _reasons(judged) == [PushTerminationReason.NONE, PushTerminationReason.FALL]


def test_success_needs_the_box_held_in_the_retreat_area(inputs: TeamJudgeInputs) -> None:
    in_retreat = dataclasses.replace(inputs, box_xy_m=torch.tensor([[3.8, 0.1]] * NUM_ENVS))

    almost = dataclasses.replace(in_retreat, hold_steps=torch.tensor([8, 9]))
    judgement = judge_team_step(almost, RULES)

    assert judgement.hold_steps.tolist() == [9, 10]
    assert judgement.reasons.tolist() == [PushTerminationReason.NONE, PushTerminationReason.SUCCESS]


def test_leaving_the_retreat_area_resets_the_hold(inputs: TeamJudgeInputs) -> None:
    judgement = judge_team_step(dataclasses.replace(inputs, hold_steps=torch.tensor([5, 5])), RULES)

    assert judgement.hold_steps.tolist() == [0, 0]


def test_robot_outside_the_arena_ends_the_episode(inputs: TeamJudgeInputs) -> None:
    outside = (torch.tensor([[0.8, 0.25], [0.8, 4.5]]), inputs.robots_xy_m[1])

    assert _reasons(dataclasses.replace(inputs, robots_xy_m=outside)) == [
        PushTerminationReason.NONE, PushTerminationReason.OUT_OF_BOUNDS,
    ]  # fmt: skip


def test_time_out_at_the_episode_length(inputs: TeamJudgeInputs) -> None:
    timed_out = dataclasses.replace(inputs, episode_steps=torch.tensor([599, 600]))

    assert _reasons(timed_out) == [PushTerminationReason.NONE, PushTerminationReason.TIME_OUT]
