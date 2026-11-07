import dataclasses
import math

import pytest
import torch

from cooperative_go2.entities.partner_channel import PartnerMessage, PartnerReception
from cooperative_go2.entities.push_config import PushObservationScales
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.team_marl_observation import (
    TEAM_ACTOR_OBSERVATION_DIM,
    TEAM_ACTOR_OBSERVATION_LAYOUT,
    TeamActorObservationInputs,
    TeamRobotTruth,
    TeamStateInputs,
    build_team_actor_observation,
    build_team_state,
    team_state_dim,
)

NUM_ENVS = 2
MAX_AGE_S = 1.0
SCALES = PushObservationScales(position=1.0, lin_vel=2.0, ang_vel=0.25)


def _slice(name: str) -> slice:
    start = 0
    for component, size in TEAM_ACTOR_OBSERVATION_LAYOUT:
        if component == name:
            return slice(start, start + size)
        start += size
    raise KeyError(name)


def _planar(x_m: float, y_m: float, yaw_rad: float = 0.0) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x_m, y_m]] * NUM_ENVS),
        yaw_rad=torch.full((NUM_ENVS,), yaw_rad),
        lin_vel_xy_mps=torch.zeros(NUM_ENVS, 2),
        yaw_rate_radps=torch.zeros(NUM_ENVS),
    )


def _reception(x_m: float, y_m: float, received: bool, age_s: float = 0.1) -> PartnerReception:
    return PartnerReception(
        message=PartnerMessage(
            position_xy_m=torch.tensor([[x_m, y_m]] * NUM_ENVS),
            yaw_rad=torch.zeros(NUM_ENVS),
            command=torch.tensor([[0.5, 0.0, 0.2]] * NUM_ENVS),
            stamp_s=torch.zeros(NUM_ENVS),
        ),
        received=torch.full((NUM_ENVS,), received),
        age_s=torch.full((NUM_ENVS,), age_s if received else MAX_AGE_S),
    )


@pytest.fixture
def actor_inputs() -> TeamActorObservationInputs:
    # 自分は原点で +y を向き、箱は +x に 1 m、相手は自分の右（-y）に 0.5 m
    return TeamActorObservationInputs(
        robot=_planar(0.0, 0.0, yaw_rad=math.pi / 2),
        robot_projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * NUM_ENVS),
        box=_planar(1.0, 0.0),
        goal_xy_m=torch.tensor([[3.0, 0.0]] * NUM_ENVS),
        last_commands=torch.zeros(NUM_ENVS, 3),
        side_sign=1.0,
        partner=_reception(0.0, -0.5, received=True),
    )


def _truth(x_m: float) -> TeamRobotTruth:
    return TeamRobotTruth(
        body=_planar(x_m, 0.0),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * NUM_ENVS),
        last_commands=torch.zeros(NUM_ENVS, 3),
        box_contact=torch.zeros(NUM_ENVS, dtype=torch.bool),
    )


@pytest.fixture
def state_inputs() -> TeamStateInputs:
    return TeamStateInputs(
        robots=(_truth(0.0), _truth(0.5)),
        box=_planar(1.4, 0.0),
        goal_xy_m=torch.tensor([[3.8, 0.0]] * NUM_ENVS),
        episode_progress=torch.zeros(NUM_ENVS),
    )


def test_actor_observation_has_the_layout_dimension(actor_inputs: TeamActorObservationInputs) -> None:
    observation = build_team_actor_observation(actor_inputs, SCALES, MAX_AGE_S)

    assert observation.shape == (NUM_ENVS, TEAM_ACTOR_OBSERVATION_DIM)


def test_partner_position_is_expressed_in_own_robot_frame(actor_inputs: TeamActorObservationInputs) -> None:
    observation = build_team_actor_observation(actor_inputs, SCALES, MAX_AGE_S)

    # +y を向いた自分から見て、ワールドの -y に 0.5 m の相手は真後ろ（ロボット座標系の -x）
    assert observation[0, _slice("partner_rel_pos_xy")].tolist() == pytest.approx([-0.5, 0.0], abs=1e-6)
    assert observation[0, _slice("partner_received")].item() == 1.0
    assert observation[0, _slice("partner_age")].item() == pytest.approx(0.1)


def test_partner_fields_are_zero_until_a_message_is_received(actor_inputs: TeamActorObservationInputs) -> None:
    inputs = dataclasses.replace(actor_inputs, partner=_reception(0.0, -0.5, received=False))

    observation = build_team_actor_observation(inputs, SCALES, MAX_AGE_S)

    for name in ("partner_rel_pos_xy", "partner_rel_yaw_sin_cos", "partner_command", "partner_received"):
        assert torch.all(observation[:, _slice(name)] == 0.0), name
    assert torch.all(observation[:, _slice("partner_age")] == 1.0)


def test_side_sign_distinguishes_left_and_right_actors(actor_inputs: TeamActorObservationInputs) -> None:
    left = build_team_actor_observation(actor_inputs, SCALES, MAX_AGE_S)
    right = build_team_actor_observation(dataclasses.replace(actor_inputs, side_sign=-1.0), SCALES, MAX_AGE_S)

    assert left[0, _slice("side_sign")].item() == 1.0
    assert right[0, _slice("side_sign")].item() == -1.0


def test_actor_observation_inputs_do_not_carry_global_truth() -> None:
    # Actor の材料の型に、箱との接触・相手の真の現在の状態・エピソードの経過が入っていないこと
    names = {field.name for field in dataclasses.fields(TeamActorObservationInputs)}

    assert names == {
        "robot", "robot_projected_gravity", "box", "goal_xy_m", "last_commands", "side_sign", "partner",
    }  # fmt: skip


def test_state_has_the_team_state_dimension(state_inputs: TeamStateInputs) -> None:
    state = build_team_state(state_inputs, own_side_sign=1.0, scales=SCALES)

    assert state.shape == (NUM_ENVS, team_state_dim(2))


def test_state_differs_between_actors_only_in_the_own_side_sign(state_inputs: TeamStateInputs) -> None:
    left = build_team_state(state_inputs, own_side_sign=1.0, scales=SCALES)
    right = build_team_state(state_inputs, own_side_sign=-1.0, scales=SCALES)

    assert torch.equal(left[:, :-1], right[:, :-1])
    assert left[0, -1].item() == 1.0 and right[0, -1].item() == -1.0


def test_state_reflects_the_current_true_partner_position(state_inputs: TeamStateInputs) -> None:
    moved = dataclasses.replace(state_inputs, robots=(_truth(0.0), _truth(0.9)))

    before = build_team_state(state_inputs, own_side_sign=1.0, scales=SCALES)
    after = build_team_state(moved, own_side_sign=1.0, scales=SCALES)

    assert not torch.equal(before, after)
