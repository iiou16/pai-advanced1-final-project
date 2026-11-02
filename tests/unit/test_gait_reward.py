import pytest
import torch

from cooperative_go2.entities.gait_action import GaitActionHistory
from cooperative_go2.entities.gait_config import GaitRewardConfig, GaitRewardScales
from cooperative_go2.entities.gait_reward import GaitRewardInputs, compute_gait_reward_terms

NUM_ENVS = 2
FORWARD_COMMAND = torch.tensor([[0.5, 0.0, 0.0]] * NUM_ENVS)


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
        base_lin_vel=FORWARD_COMMAND.clone(),
        base_pos_m=torch.tensor([[0.0, 0.0, 0.3]] * NUM_ENVS),
    )
    return GaitRewardInputs(
        state=state,
        commands=FORWARD_COMMAND,
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
    inputs = GaitRewardInputs(standing, FORWARD_COMMAND, ideal_inputs.actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    # 指令 0.5 m/s に対して静止している: 誤差^2 = 0.25 → exp(-0.25 / 0.25)
    assert terms.tracking_lin_vel[0].item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())


def test_action_rate_penalizes_change_from_previous_action(ideal_inputs, reward_config) -> None:
    actions = GaitActionHistory(current=torch.full((NUM_ENVS, 12), 0.5), previous=torch.zeros(NUM_ENVS, 12))
    inputs = GaitRewardInputs(ideal_inputs.state, FORWARD_COMMAND, actions, ideal_inputs.default_dof_pos)

    terms = compute_gait_reward_terms(inputs, reward_config)

    assert terms.action_rate[0].item() == pytest.approx(12 * 0.25)


def test_weighted_terms_multiply_scale_and_control_period(ideal_inputs, reward_config) -> None:
    terms = compute_gait_reward_terms(ideal_inputs, reward_config)

    weighted = terms.weighted(reward_config.scales, control_dt_s=0.02)

    assert weighted.tracking_lin_vel[0].item() == pytest.approx(0.02)
    assert weighted.tracking_ang_vel[0].item() == pytest.approx(0.004)


def test_total_sums_all_terms(ideal_inputs, reward_config) -> None:
    terms = compute_gait_reward_terms(ideal_inputs, reward_config).weighted(reward_config.scales, control_dt_s=0.02)

    # 追従の 2 項だけが残る: (1.0 + 0.2) × 0.02
    assert torch.allclose(terms.total(), torch.full((NUM_ENVS,), 0.024))
