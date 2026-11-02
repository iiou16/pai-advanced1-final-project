import pytest
import torch

from cooperative_go2.entities.gait_config import ValueRange, VelocityCommandConfig
from cooperative_go2.entities.velocity_command import VelocityCommand, VelocityCommandSampler

SEED = 7


def _config(zero_lin_vel_probability: float = 0.0, zero_yaw_rate_probability: float = 0.0) -> VelocityCommandConfig:
    return VelocityCommandConfig(
        lin_vel_x_mps=ValueRange(low=-1.0, high=1.0),
        lin_vel_y_mps=ValueRange(low=0.0, high=0.0),
        yaw_rate_radps=ValueRange(low=-2.0, high=2.0),
        zero_lin_vel_probability=zero_lin_vel_probability,
        zero_yaw_rate_probability=zero_yaw_rate_probability,
        resampling_time_s=4.0,
    )


def _sampler(num_envs: int, config: VelocityCommandConfig | None = None) -> VelocityCommandSampler:
    return VelocityCommandSampler(config or _config(), num_envs, torch.Generator().manual_seed(SEED))


def test_resampled_commands_stay_within_configured_ranges() -> None:
    sampler = _sampler(num_envs=256)

    sampler.resample(torch.ones(256, dtype=torch.bool))

    commands = sampler.commands
    assert commands[:, 0].min() >= -1.0 and commands[:, 0].max() <= 1.0
    assert torch.equal(commands[:, 1], torch.zeros(256))
    assert commands[:, 2].min() >= -2.0 and commands[:, 2].max() <= 2.0


def test_resampled_commands_spread_over_the_range() -> None:
    sampler = _sampler(num_envs=256)

    sampler.resample(torch.ones(256, dtype=torch.bool))

    vx = sampler.commands[:, 0]
    assert vx.min() < -0.8 and vx.max() > 0.8


def test_commands_are_zero_before_first_resample() -> None:
    assert torch.equal(_sampler(num_envs=2).commands, torch.zeros(2, 3))


def test_same_seed_gives_same_commands() -> None:
    first, second = _sampler(num_envs=4), _sampler(num_envs=4)

    first.resample(torch.ones(4, dtype=torch.bool))
    second.resample(torch.ones(4, dtype=torch.bool))

    assert torch.equal(first.commands, second.commands)


def test_only_masked_envs_are_resampled() -> None:
    sampler = _sampler(num_envs=3)
    sampler.resample(torch.ones(3, dtype=torch.bool))
    before = sampler.commands.clone()

    sampler.resample(torch.tensor([False, True, False]))

    assert torch.equal(sampler.commands[0], before[0])
    assert torch.equal(sampler.commands[2], before[2])
    assert not torch.equal(sampler.commands[1], before[1])


def test_fixed_command_is_applied_to_all_resampled_envs() -> None:
    sampler = _sampler(num_envs=2)

    sampler.fix(VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.5))
    sampler.resample(torch.ones(2, dtype=torch.bool))

    assert torch.equal(sampler.commands, torch.tensor([[0.0, 0.0, 1.5]] * 2))


def test_fixing_replaces_current_commands_immediately() -> None:
    sampler = _sampler(num_envs=2)
    sampler.resample(torch.ones(2, dtype=torch.bool))

    sampler.fix(VelocityCommand(vx_mps=0.3, vy_mps=0.1, yaw_rate_radps=-0.5))

    assert torch.allclose(sampler.commands, torch.tensor([[0.3, 0.1, -0.5]] * 2))


def test_certain_zero_probabilities_give_standstill_commands() -> None:
    sampler = _sampler(num_envs=8, config=_config(zero_lin_vel_probability=1.0, zero_yaw_rate_probability=1.0))

    sampler.resample(torch.ones(8, dtype=torch.bool))

    assert torch.equal(sampler.commands, torch.zeros(8, 3))


def test_zero_lin_vel_keeps_yaw_rate_for_turning_in_place() -> None:
    sampler = _sampler(num_envs=8, config=_config(zero_lin_vel_probability=1.0))

    sampler.resample(torch.ones(8, dtype=torch.bool))

    assert torch.equal(sampler.commands[:, :2], torch.zeros(8, 2))
    assert (sampler.commands[:, 2] != 0.0).all()


@pytest.mark.parametrize(
    ("zero_lin_vel_probability", "zero_yaw_rate_probability"),
    [(0.3, 0.0), (0.0, 0.3)],
    ids=["zero-lin-vel", "zero-yaw-rate"],
)
def test_components_are_zeroed_at_configured_rate(
    zero_lin_vel_probability: float, zero_yaw_rate_probability: float
) -> None:
    num_envs = 4000
    config = _config(zero_lin_vel_probability, zero_yaw_rate_probability)
    sampler = _sampler(num_envs, config)

    sampler.resample(torch.ones(num_envs, dtype=torch.bool))

    zero_vx_rate = (sampler.commands[:, 0] == 0.0).float().mean().item()
    zero_yaw_rate_rate = (sampler.commands[:, 2] == 0.0).float().mean().item()
    assert zero_vx_rate == pytest.approx(zero_lin_vel_probability, abs=0.03)
    assert zero_yaw_rate_rate == pytest.approx(zero_yaw_rate_probability, abs=0.03)
