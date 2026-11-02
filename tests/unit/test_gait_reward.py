import dataclasses

import pytest
import torch

from cooperative_go2.entities.gait_action import GaitActionHistory
from cooperative_go2.entities.gait_config import GaitRewardConfig, GaitRewardScales
from cooperative_go2.entities.gait_reward import GAIT_REWARD_TERM_NAMES, GaitRewardInputs, compute_gait_reward_terms

NUM_ENVS = 2


def _forward_command() -> torch.Tensor:
    return torch.tensor([[0.5, 0.0, 0.0]] * NUM_ENVS)


@pytest.fixture
def reward_config() -> GaitRewardConfig:
    return GaitRewardConfig(
        tracking_sigma=0.25,
        base_height_target_m=0.3,
        scales=GaitRewardScales(
            tracking_lin_vel=1.0,
            tracking_ang_vel=0.2,
            lin_vel_z=-1.0,
            base_height=-50.0,
            action_rate=-0.005,
            similar_to_default=-0.1,
        ),
    )


@pytest.fixture
def ideal_inputs(make_go2_state) -> GaitRewardInputs:
    """指令どおりに前進し、目標の高さ・既定姿勢で、行動も変化しない理想状態。"""
    state = make_go2_state(
        num_envs=NUM_ENVS,
        base_lin_vel=_forward_command(),
        base_pos_m=torch.tensor([[0.0, 0.0, 0.3]] * NUM_ENVS),
    )
    return GaitRewardInputs(
        state=state,
        commands=_forward_command(),
        actions=GaitActionHistory.zeros(NUM_ENVS, 12, "cpu"),
        default_dof_pos=torch.zeros(12),
    )


def test_ideal_state_gets_full_tracking_reward_and_no_penalty(ideal_inputs, reward_config) -> None:
    terms = compute_gait_reward_terms(ideal_inputs, reward_config)

    assert torch.allclose(terms.tracking_lin_vel, torch.ones(NUM_ENVS))
    assert torch.allclose(terms.tracking_ang_vel, torch.ones(NUM_ENVS))
    for penalty in (terms.lin_vel_z, terms.base_height, terms.action_rate, terms.similar_to_default):
        assert torch.allclose(penalty, torch.zeros(NUM_ENVS))


def test_tracking_reward_decays_with_velocity_error(make_go2_state, ideal_inputs, reward_config) -> None:
    standing = make_go2_state(num_envs=NUM_ENVS, base_pos_m=ideal_inputs.state.base_pos_m)
    inputs = GaitRewardInputs(standing, _forward_command(), ideal_inputs.actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    # 指令 0.5 m/s に対して静止している: 誤差^2 = 0.25 → exp(-0.25 / 0.25)
    assert terms.tracking_lin_vel[0].item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())


def test_action_rate_penalizes_change_from_previous_action(ideal_inputs, reward_config) -> None:
    actions = GaitActionHistory(current=torch.full((NUM_ENVS, 12), 0.5), previous=torch.zeros(NUM_ENVS, 12))
    inputs = GaitRewardInputs(ideal_inputs.state, _forward_command(), actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    assert terms.action_rate[0].item() == pytest.approx(12 * 0.25)


def test_total_sums_all_terms(ideal_inputs, reward_config) -> None:
    terms = compute_gait_reward_terms(ideal_inputs, reward_config).weighted(reward_config.scales, control_dt_s=0.02)

    # 追従の 2 項だけが残る: (1.0 + 0.2) × 0.02
    assert torch.allclose(terms.total(), torch.full((NUM_ENVS,), 0.024))


def test_penalty_terms_measure_their_own_quantities(make_go2_state, ideal_inputs, reward_config) -> None:
    state = make_go2_state(
        num_envs=NUM_ENVS,
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.4]] * NUM_ENVS),
        base_pos_m=torch.tensor([[0.0, 0.0, 0.2]] * NUM_ENVS),
        dof_pos=torch.tensor([[0.1, -0.2] + [0.0] * 10] * NUM_ENVS),
    )
    inputs = GaitRewardInputs(state, _forward_command(), ideal_inputs.actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    assert terms.lin_vel_z[0].item() == pytest.approx(0.16)
    assert terms.base_height[0].item() == pytest.approx(0.01)
    # 既定姿勢からの差は符号によらず絶対値で足す
    assert terms.similar_to_default[0].item() == pytest.approx(0.3)


def test_lateral_velocity_error_reduces_tracking_reward(make_go2_state, ideal_inputs, reward_config) -> None:
    sideways = make_go2_state(num_envs=NUM_ENVS, base_lin_vel=torch.tensor([[0.5, 0.5, 0.0]] * NUM_ENVS))
    inputs = GaitRewardInputs(sideways, _forward_command(), ideal_inputs.actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    assert terms.tracking_lin_vel[0].item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())


def test_yaw_tracking_uses_only_yaw_rate(make_go2_state, ideal_inputs, reward_config) -> None:
    yaw_command = torch.tensor([[0.0, 0.0, 1.0]] * NUM_ENVS)
    roll_pitch_only = make_go2_state(num_envs=NUM_ENVS, base_ang_vel=torch.tensor([[1.0, 1.0, 0.0]] * NUM_ENVS))
    inputs = GaitRewardInputs(roll_pitch_only, yaw_command, ideal_inputs.actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    # yaw_rate は 0 なので誤差は 1。ロールとピッチの角速度は追従に影響しない
    assert terms.tracking_ang_vel[0].item() == pytest.approx(torch.exp(torch.tensor(-4.0)).item())


def test_negative_scale_turns_penalty_into_negative_reward(make_go2_state, ideal_inputs, reward_config) -> None:
    low_body = make_go2_state(num_envs=NUM_ENVS, base_pos_m=torch.tensor([[0.0, 0.0, 0.2]] * NUM_ENVS))
    inputs = GaitRewardInputs(low_body, _forward_command(), ideal_inputs.actions, ideal_inputs.default_dof_pos)

    weighted = compute_gait_reward_terms(inputs, reward_config).weighted(reward_config.scales, control_dt_s=0.02)

    # (0.2 - 0.3)^2 × (-50) × 0.02
    assert weighted.base_height[0].item() == pytest.approx(-0.01)


def test_zeroed_where_clears_every_term_of_masked_envs(ideal_inputs, reward_config) -> None:
    terms = compute_gait_reward_terms(ideal_inputs, reward_config)

    zeroed = terms.zeroed_where(torch.tensor([True, False]))

    assert zeroed.total()[0].item() == 0.0
    assert zeroed.total()[1].item() == pytest.approx(terms.total()[1].item())


def test_reward_scales_and_terms_have_the_same_fields() -> None:
    assert {field.name for field in dataclasses.fields(GaitRewardScales)} == set(GAIT_REWARD_TERM_NAMES)
