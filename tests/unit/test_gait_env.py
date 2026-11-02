"""歩行環境の 1 ステップの流れを、偽物のシミュレーションで CPU だけで確かめる。"""

import dataclasses
import math

import pytest
import torch

from cooperative_go2.entities.gait_termination import TerminationReason
from cooperative_go2.interface_adapters.config_loader import GaitSettings
from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_LAYOUT
from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.interface_adapters.gait_env import Go2GaitEnv
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP

NUM_ENVS = 2
ROLL_FALLEN_DEG = 30.0


@pytest.fixture
def env(small_gait_settings: GaitSettings, fake_simulation) -> Go2GaitEnv:
    assert small_gait_settings.gait.env.num_envs == NUM_ENVS
    return Go2GaitEnv(small_gait_settings.gait, fake_simulation, torch.Generator().manual_seed(0))


def _observation_slice(component: str) -> slice:
    start = 0
    for name, size in GAIT_OBSERVATION_LAYOUT:
        if name == component:
            return slice(start, start + size)
        start += size
    raise KeyError(component)


def _zero_actions() -> torch.Tensor:
    return torch.zeros(NUM_ENVS, 12)


def test_diverged_env_gets_zero_reward_even_if_state_is_nan(env, fake_simulation) -> None:
    nan_velocity = torch.tensor([[float("nan")] * 3, [0.5, 0.0, 0.0]])
    fake_simulation.next_state = dataclasses.replace(fake_simulation.next_state, base_lin_vel=nan_velocity)
    fake_simulation.next_diverged = torch.tensor([True, False])

    outcome = env.step_with_outcome(_zero_actions())

    assert outcome.reward[0].item() == 0.0
    assert torch.isfinite(outcome.reward).all()
    assert outcome.termination_reasons.tolist() == [int(TerminationReason.SIM_DIVERGED), int(TerminationReason.NONE)]
    assert torch.isfinite(outcome.observation).all()


def test_time_out_happens_one_step_after_episode_length(env) -> None:
    episode_steps = env.max_episode_length

    outcomes = [env.step_with_outcome(_zero_actions()) for _ in range(episode_steps + 1)]

    assert episode_steps == 5
    assert not outcomes[episode_steps - 1].done.any()
    assert outcomes[episode_steps].time_out.all()


def test_fallen_env_alone_is_reset(env, fake_simulation) -> None:
    env.step_with_outcome(torch.ones(NUM_ENVS, 12))
    tilted = torch.tensor([[ROLL_FALLEN_DEG, 0.0, 0.0], [0.0, 0.0, 0.0]])
    fake_simulation.next_state = dataclasses.replace(fake_simulation.next_state, base_euler_deg=tilted)

    outcome = env.step_with_outcome(torch.ones(NUM_ENVS, 12))

    assert outcome.done.tolist() == [True, False]
    assert env.episode_length_buf.tolist() == [0, 2]
    assert torch.equal(fake_simulation.reset_masks[-1], torch.tensor([True, False]))
    # 観測の末尾 12 次元は前回の行動。リセットした環境だけ 0 に戻る
    assert torch.equal(outcome.observation[0, -12:], torch.zeros(12))
    assert torch.equal(outcome.observation[1, -12:], torch.ones(12))


def test_latency_applies_previous_action_as_joint_target(env, fake_simulation) -> None:
    env.step_with_outcome(torch.ones(NUM_ENVS, 12))
    env.step_with_outcome(torch.full((NUM_ENVS, 12), 2.0))

    action_scale = env.cfg.env.action_scale
    assert torch.equal(fake_simulation.applied_targets[0], torch.zeros(NUM_ENVS, 12))
    assert torch.allclose(fake_simulation.applied_targets[1], torch.full((NUM_ENVS, 12), action_scale))


def test_rsl_rl_step_reports_episode_log_only_when_an_episode_ends(env, fake_simulation) -> None:
    _, _, _, extras_running = env.step(_zero_actions())
    fake_simulation.next_state = dataclasses.replace(
        fake_simulation.next_state, base_euler_deg=torch.tensor([[ROLL_FALLEN_DEG, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    _, _, dones, extras_ended = env.step(_zero_actions())

    assert "episode" not in extras_running
    assert "episode" in extras_ended
    assert extras_ended["episode"]["termination_fall"].item() == 1.0
    assert extras_ended["time_outs"].dtype == torch.float32
    assert dones.tolist() == [True, False]


def test_manual_reset_discards_partial_episode_rewards(env, fake_simulation) -> None:
    # リセット前は指令どおりに前進（追従の報酬が最大）、リセット後は静止して転倒する
    walking = dataclasses.replace(fake_simulation.next_state, base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]] * NUM_ENVS))
    fake_simulation.next_state = walking
    for _ in range(3):
        env.step_with_outcome(_zero_actions())
    env.reset()
    fake_simulation.next_state = dataclasses.replace(
        walking, base_lin_vel=torch.zeros(NUM_ENVS, 3), base_euler_deg=torch.full((NUM_ENVS, 3), ROLL_FALLEN_DEG)
    )

    outcome = env.step_with_outcome(_zero_actions())

    # 集計はリセット後の 1 ステップだけ: 指令 0.5 m/s に静止 → exp(-0.25 / 0.25) × 係数（1 秒あたり）
    assert outcome.episode_summary is not None
    reward_config = env.cfg.reward
    expected = math.exp(-0.25 / reward_config.tracking_sigma) * reward_config.scales.tracking_lin_vel
    assert outcome.episode_summary.reward_per_second["tracking_lin_vel"].item() == pytest.approx(expected)


def test_command_fixed_mid_episode_reaches_next_observation_immediately(env) -> None:
    command_slice = _observation_slice("velocity_command")
    ang_vel_scale = env.cfg.observation.ang_vel

    env.fix_command(VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.0))

    observation = env.get_observations()[POLICY_OBSERVATION_GROUP]
    expected = torch.tensor([0.0, 0.0, 1.0 * ang_vel_scale])
    assert torch.allclose(observation[:, command_slice], expected.expand(NUM_ENVS, -1))


@pytest.fixture
def noisy_env(small_gait_settings: GaitSettings, fake_simulation) -> Go2GaitEnv:
    config = small_gait_settings.gait
    noisy_config = dataclasses.replace(config, env=dataclasses.replace(config.env, reset_joint_pos_noise_rad=0.1))
    return Go2GaitEnv(noisy_config, fake_simulation, torch.Generator().manual_seed(0))


def test_reset_perturbs_joint_angles_within_noise_width(noisy_env, fake_simulation) -> None:
    reset_dof_pos = fake_simulation.reset_dof_pos[-1]

    offsets = reset_dof_pos - fake_simulation.default_dof_pos
    assert offsets.abs().max() <= 0.1
    assert not torch.equal(reset_dof_pos[0], reset_dof_pos[1])


def test_observation_after_reset_reflects_perturbed_joint_angles(noisy_env, fake_simulation) -> None:
    joint_pos_offset_slice = _observation_slice("joint_pos_offset")

    observation = noisy_env.get_observations()[POLICY_OBSERVATION_GROUP]

    expected = (fake_simulation.reset_dof_pos[-1] - fake_simulation.default_dof_pos) * noisy_env.cfg.observation.dof_pos
    assert torch.allclose(observation[:, joint_pos_offset_slice], expected)
