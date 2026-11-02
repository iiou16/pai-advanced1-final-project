import torch

from cooperative_go2.entities.gait_config import ValueRange, VelocityCommandConfig
from cooperative_go2.entities.velocity_command import VelocityCommand, VelocityCommandSampler

SEED = 7


def _config() -> VelocityCommandConfig:
    return VelocityCommandConfig(
        lin_vel_x_mps=ValueRange(low=-1.0, high=1.0),
        lin_vel_y_mps=ValueRange(low=0.0, high=0.0),
        yaw_rate_radps=ValueRange(low=-2.0, high=2.0),
        resampling_time_s=4.0,
    )


def _sampler(num_envs: int) -> VelocityCommandSampler:
    return VelocityCommandSampler(_config(), num_envs, torch.Generator().manual_seed(SEED))


def test_resampled_commands_stay_within_configured_ranges() -> None:
    sampler = _sampler(num_envs=256)

    sampler.resample(torch.ones(256, dtype=torch.bool))

    commands = sampler.commands
    assert commands[:, 0].min() >= -1.0 and commands[:, 0].max() <= 1.0
    assert torch.equal(commands[:, 1], torch.zeros(256))
    assert commands[:, 2].min() >= -2.0 and commands[:, 2].max() <= 2.0


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
