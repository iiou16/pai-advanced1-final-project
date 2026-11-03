"""評価用のラッパーが、発散と転倒を評価のポートの形に変換することの確認（偽物のシミュレーションを使う）。"""

import dataclasses

import pytest
import torch

from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.interface_adapters.config_loader import GaitSettings
from cooperative_go2.interface_adapters.gait_env import Go2GaitEnv
from cooperative_go2.interface_adapters.gait_rollout import Go2GaitRolloutEnvironment
from cooperative_go2.use_cases.simulation_errors import SimulationDivergedError

NUM_ENVS = 2
ROLL_FALLEN_DEG = 30.0
TURN = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.0)


@pytest.fixture
def rollout(small_gait_settings: GaitSettings, fake_simulation) -> Go2GaitRolloutEnvironment:
    env = Go2GaitEnv(small_gait_settings.gait, fake_simulation, torch.Generator().manual_seed(0))
    return Go2GaitRolloutEnvironment(env)


def test_reset_fixes_command_into_observation(rollout) -> None:
    observation = rollout.reset_with_command(TURN)

    # 観測の 7〜9 番目は速度指令。yaw_rate には ang_vel の正規化係数が掛かる
    assert observation[:, 6:8].abs().sum().item() == 0.0
    assert (observation[:, 8] > 0).all()


def test_simulator_reported_divergence_stops_evaluation(rollout, fake_simulation) -> None:
    rollout.reset_with_command(TURN)
    fake_simulation.next_diverged = torch.tensor([False, True])

    with pytest.raises(SimulationDivergedError):
        rollout.step(torch.zeros(NUM_ENVS, 12))


def test_non_finite_velocity_stops_evaluation_even_if_not_reported(rollout, fake_simulation) -> None:
    rollout.reset_with_command(TURN)
    nan_ang_vel = torch.tensor([[0.0, 0.0, float("nan")], [0.0, 0.0, 0.0]])
    proprioception = dataclasses.replace(fake_simulation.next_state.proprioception, base_ang_vel=nan_ang_vel)
    fake_simulation.next_state = dataclasses.replace(fake_simulation.next_state, proprioception=proprioception)

    with pytest.raises(SimulationDivergedError):
        rollout.step(torch.zeros(NUM_ENVS, 12))


def test_only_fall_is_reported_as_fallen(rollout, fake_simulation) -> None:
    rollout.reset_with_command(TURN)
    fake_simulation.next_state = dataclasses.replace(
        fake_simulation.next_state, base_euler_deg=torch.tensor([[ROLL_FALLEN_DEG, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )

    step = rollout.step(torch.zeros(NUM_ENVS, 12))

    assert step.fallen.tolist() == [True, False]
