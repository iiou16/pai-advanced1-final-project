import dataclasses

import pytest
import torch
from conftest import TEAM_NUM_ENVS, FakeTeamSimulation, TeamMarlEnvFactory

from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.team_marl_observation import (
    TEAM_ACTOR_OBSERVATION_DIM,
    TEAM_ACTOR_OBSERVATION_LAYOUT,
    TEAM_STATE_ROBOT_LAYOUT,
    team_state_dim,
)
from cooperative_go2.entities.team_task_config import sample_team_start_positions
from cooperative_go2.interface_adapters.team_marl_env import TeamSizeError

STAND_STILL = (torch.zeros(TEAM_NUM_ENVS, 3), torch.zeros(TEAM_NUM_ENVS, 3))


def _slice(name: str) -> slice:
    start = 0
    for component, size in TEAM_ACTOR_OBSERVATION_LAYOUT:
        if component == name:
            return slice(start, start + size)
        start += size
    raise KeyError(name)


def _move_robot(simulation: FakeTeamSimulation, index: int, x_m: float, y_m: float) -> None:
    robot = simulation.robots[index]
    simulation.robots[index] = dataclasses.replace(robot, position_xy_m=torch.tensor([[x_m, y_m]] * TEAM_NUM_ENVS))


def test_observations_states_and_rewards_have_one_entry_per_robot(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()

    outcome = env.step(STAND_STILL)

    assert len(outcome.observations) == 2 and len(outcome.states) == 2
    for observation, state in zip(outcome.observations, outcome.states, strict=True):
        assert observation.shape == (TEAM_NUM_ENVS, TEAM_ACTOR_OBSERVATION_DIM)
        assert state.shape == (TEAM_NUM_ENVS, team_state_dim(2))
    assert outcome.rewards.shape == (2, TEAM_NUM_ENVS)
    assert outcome.done.shape == (TEAM_NUM_ENVS,)


def test_actions_for_a_different_number_of_robots_are_rejected(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()

    with pytest.raises(TeamSizeError):
        env.step((torch.zeros(TEAM_NUM_ENVS, 3),))


def test_no_partner_message_is_observed_right_after_reset(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()

    for observation in env.observations:
        assert torch.all(observation[:, _slice("partner_received")] == 0.0)


def test_actor_sees_the_partner_as_sent_in_the_previous_step_not_its_true_current_state(
    make_team_marl_env: TeamMarlEnvFactory,
) -> None:
    env, simulation = make_team_marl_env()
    _move_robot(simulation, 0, 0.7, 0.25)
    _move_robot(simulation, 1, 0.7, -0.25)
    env.reset()
    # 物理を進める間に、相手（go2_b）の真の位置を大きく動かす
    simulation.step_hook = lambda: _move_robot(simulation, 1, 0.2, -0.25)
    sent_partner_xy = simulation.robots[1].position_xy_m.clone()

    outcome = env.step(STAND_STILL)

    observation = outcome.observations[0]
    own_xy = simulation.robots[0].position_xy_m
    expected_rel = sent_partner_xy - own_xy
    assert torch.allclose(observation[:, _slice("partner_rel_pos_xy")], expected_rel)
    assert torch.all(observation[:, _slice("partner_received")] == 1.0)
    # Critic の状態は、相手の真の現在の位置（x = 0.2）を持つ。2 台目の成分は 1 台目の成分のすぐ後ろ
    partner_x_index = sum(size for _, size in TEAM_STATE_ROBOT_LAYOUT)
    assert torch.allclose(outcome.states[0][:, partner_x_index], torch.full((TEAM_NUM_ENVS,), 0.2))


def test_shared_terms_give_both_robots_the_same_box_progress(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, simulation = make_team_marl_env()

    def push_box() -> None:
        simulation.box = dataclasses.replace(
            simulation.box, position_xy_m=simulation.box.position_xy_m + torch.tensor([0.01, 0.0])
        )

    simulation.step_hook = push_box
    outcome = env.step(STAND_STILL)

    progress = outcome.reward_terms.box_progress
    assert torch.equal(progress[0], progress[1])
    assert torch.all(progress[0] > 0)


def test_a_fall_of_one_robot_ends_and_resets_the_episode_for_both(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, simulation = make_team_marl_env()
    env.step(STAND_STILL)

    def tilt_go2_b_in_env_1() -> None:
        simulation.euler_deg[1] = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

    simulation.step_hook = tilt_go2_b_in_env_1
    outcome = env.step(STAND_STILL)

    assert outcome.done.tolist() == [False, True]
    assert outcome.termination_reasons[1] == int(PushTerminationReason.FALL)
    assert outcome.reward_terms.robot_fall[:, 1].tolist() == [0.0, -10.0]
    assert env.episode_length_buf.tolist() == [2, 0]
    # リセットした環境では、前のエピソードに届いた相手の状態を使わない
    assert outcome.observations[0][:, _slice("partner_received")].flatten().tolist() == [1.0, 0.0]
    assert outcome.observations[0][:, _slice("last_command")].abs().sum() == 0.0


def test_time_out_after_the_episode_length(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()

    reasons = [env.step(STAND_STILL).termination_reasons for _ in range(env.max_episode_length)]

    assert all(int(reason) == int(PushTerminationReason.NONE) for reason in reasons[-2])
    assert all(int(reason) == int(PushTerminationReason.TIME_OUT) for reason in reasons[-1])


def test_first_reset_draws_the_same_start_positions_as_the_rule_based_check(
    make_team_marl_env: TeamMarlEnvFactory,
) -> None:
    env, simulation = make_team_marl_env(seed=5001)
    generator = torch.Generator().manual_seed(5001)
    initial_xy = ((0.7, 0.25), (0.7, -0.25))

    expected = sample_team_start_positions(initial_xy, env.cfg.reset, TEAM_NUM_ENVS, generator)

    _, first_positions = simulation.reset_calls[0]
    for drawn, reference in zip(first_positions, expected, strict=True):
        assert torch.equal(drawn, reference)


def test_leaving_the_arena_ends_the_episode_and_penalizes_the_robot_that_left(
    make_team_marl_env: TeamMarlEnvFactory,
) -> None:
    env, simulation = make_team_marl_env()
    simulation.step_hook = lambda: _move_robot(simulation, 0, -5.5, 0.25)

    outcome = env.step(STAND_STILL)

    assert outcome.termination_reasons.tolist() == [int(PushTerminationReason.OUT_OF_BOUNDS)] * TEAM_NUM_ENVS
    assert outcome.reward_terms.out_of_bounds[:, 0].tolist() == [-10.0, 0.0]


def test_approach_reward_is_zero_for_robots_that_stay_still_after_reset(
    make_team_marl_env: TeamMarlEnvFactory,
) -> None:
    env, simulation = make_team_marl_env()

    first = env.step(STAND_STILL)
    simulation.step_hook = lambda: _move_robot(simulation, 0, 0.9, 0.25)
    second = env.step(STAND_STILL)

    assert torch.all(first.reward_terms.approach_progress == 0.0)
    assert torch.all(second.reward_terms.approach_progress[0] != 0.0)
    assert torch.all(second.reward_terms.approach_progress[1] == 0.0)
