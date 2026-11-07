import dataclasses

import pytest
import torch

from cooperative_go2.entities.team_marl_config import TeamMarlRewardConfig, TeamMarlRewardScales
from cooperative_go2.entities.team_marl_reward import (
    TeamMarlRewardInputs,
    compute_team_marl_reward_terms,
    push_position_distances_m,
    robots_too_close,
)

NUM_ENVS = 2
SCALES = TeamMarlRewardScales(
    box_progress=10.0, success=10.0, box_rotation=-1.0, approach_progress=2.0, intended_face_contact=0.02,
    robot_fall=-10.0, out_of_bounds=-10.0, robot_collision=-1.0, control_effort=-0.01,
)  # fmt: skip
CONFIG = TeamMarlRewardConfig(
    intended_face_min_cos=0.8, min_robot_separation_m=0.35, push_standoff_m=0.6, push_lateral_offset_m=0.25,
    scales=SCALES,
)  # fmt: skip
NO_ENVS = torch.zeros(NUM_ENVS, dtype=torch.bool)


@pytest.fixture
def inputs() -> TeamMarlRewardInputs:
    # 箱は x = 1.4、退避領域は +x。2 台は箱の後ろに 0.5 m 離れて並ぶ
    return TeamMarlRewardInputs(
        previous_goal_distance_m=torch.full((NUM_ENVS,), 2.4),
        goal_distance_m=torch.full((NUM_ENVS,), 2.3),
        succeeded=NO_ENVS,
        previous_box_yaw_rad=torch.zeros(NUM_ENVS),
        box_yaw_rad=torch.full((NUM_ENVS,), 0.05),
        box_xy_m=torch.tensor([[1.4, 0.0]] * NUM_ENVS),
        goal_xy_m=torch.tensor([[3.8, 0.0]] * NUM_ENVS),
        robots_xy_m=(torch.tensor([[0.8, 0.25]] * NUM_ENVS), torch.tensor([[0.8, -0.25]] * NUM_ENVS)),
        previous_push_distances_m=(torch.full((NUM_ENVS,), 0.5), torch.full((NUM_ENVS,), 0.3)),
        push_distances_m=(torch.full((NUM_ENVS,), 0.4), torch.full((NUM_ENVS,), 0.35)),
        box_contacts=(torch.ones(NUM_ENVS, dtype=torch.bool), NO_ENVS),
        fallen=(NO_ENVS, torch.tensor([True, False])),
        out_of_bounds=(torch.tensor([False, True]), NO_ENVS),
        commands=(torch.tensor([[0.6, 0.0, 0.0]] * NUM_ENVS), torch.zeros(NUM_ENVS, 3)),
    )


def test_terms_have_one_row_per_robot(inputs: TeamMarlRewardInputs) -> None:
    terms = compute_team_marl_reward_terms(inputs, CONFIG)

    assert terms.per_robot_total().shape == (2, NUM_ENVS)


def test_shared_terms_are_identical_for_both_robots(inputs: TeamMarlRewardInputs) -> None:
    terms = compute_team_marl_reward_terms(dataclasses.replace(inputs, succeeded=torch.tensor([True, False])), CONFIG)

    for shared in (terms.box_progress, terms.success, terms.box_rotation):
        assert torch.equal(shared[0], shared[1])
    assert terms.box_progress[0].tolist() == pytest.approx([0.1, 0.1], abs=1e-6)
    assert terms.success[0].tolist() == [1.0, 0.0]
    assert terms.box_rotation[0].tolist() == pytest.approx([0.05, 0.05])


def test_individual_terms_apply_only_to_the_robot_concerned(inputs: TeamMarlRewardInputs) -> None:
    terms = compute_team_marl_reward_terms(inputs, CONFIG)

    assert terms.robot_fall.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert terms.out_of_bounds.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert terms.intended_face_contact.tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert terms.control_effort[:, 0].tolist() == pytest.approx([0.36, 0.0])
    assert terms.approach_progress[:, 0].tolist() == pytest.approx([0.1, -0.05], abs=1e-6)


def test_push_positions_are_side_by_side_behind_the_box() -> None:
    # 箱は原点、退避領域は +x。左（+1）の押す位置は (-0.6, 0.25)、右（-1）は (-0.6, -0.25)
    box = torch.zeros(1, 2)
    goal = torch.tensor([[3.0, 0.0]])
    at_left = (torch.tensor([[-0.6, 0.25]]), torch.tensor([[-0.6, 0.25]]))

    distances = push_position_distances_m(at_left, box, goal, (1.0, -1.0), CONFIG)

    assert distances[0].item() == pytest.approx(0.0, abs=1e-6)
    assert distances[1].item() == pytest.approx(0.5, abs=1e-6)


def test_push_positions_follow_the_direction_to_the_goal() -> None:
    # 退避領域が +y にあれば、押す位置は箱の -y 側に並び、左（+1）は -x 側
    distances = push_position_distances_m(
        (torch.tensor([[-0.25, -0.6]]),), torch.zeros(1, 2), torch.tensor([[0.0, 3.0]]), (1.0,), CONFIG
    )

    assert distances[0].item() == pytest.approx(0.0, abs=1e-6)


def test_robots_closer_than_the_separation_are_both_penalized() -> None:
    close = robots_too_close((torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([[0.0, 0.3], [0.0, 0.5]])), 0.35)

    assert close[0].tolist() == [True, False]
    assert close[1].tolist() == [True, False]


def test_zeroed_where_clears_every_term_for_masked_envs(inputs: TeamMarlRewardInputs) -> None:
    terms = compute_team_marl_reward_terms(inputs, CONFIG).weighted(SCALES).zeroed_where(torch.tensor([True, False]))

    totals = terms.per_robot_total()
    assert totals[:, 0].tolist() == [0.0, 0.0]
    assert totals[:, 1].abs().sum() > 0
