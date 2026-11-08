import pytest
import torch

from cooperative_go2.entities.command_delay import CommandDelayLine, InvalidCommandDelayError

NUM_ENVS = 2


def _command(value: float) -> torch.Tensor:
    return torch.full((NUM_ENVS, 3), value)


def test_zero_delay_returns_pushed_commands() -> None:
    line = CommandDelayLine(0, NUM_ENVS, "cpu")

    delivered = line.push(_command(0.5))

    assert torch.equal(delivered, _command(0.5))


def test_commands_come_out_after_delay_steps_with_stop_commands_before() -> None:
    line = CommandDelayLine(2, NUM_ENVS, "cpu")

    delivered = [line.push(_command(value)) for value in (0.1, 0.2, 0.3, 0.4)]

    assert torch.equal(delivered[0], _command(0.0))
    assert torch.equal(delivered[1], _command(0.0))
    assert torch.equal(delivered[2], _command(0.1))
    assert torch.equal(delivered[3], _command(0.2))


def test_reset_replaces_pending_commands_with_stop_only_for_masked_envs() -> None:
    line = CommandDelayLine(1, NUM_ENVS, "cpu")
    line.push(_command(0.5))

    line.reset_where(torch.tensor([True, False]))
    delivered = line.push(_command(0.9))

    assert torch.equal(delivered[0], torch.zeros(3))
    assert torch.equal(delivered[1], torch.full((3,), 0.5))


@pytest.mark.parametrize("delay_steps", [-1, 1.5, True])
def test_invalid_delay_steps_is_rejected(delay_steps: object) -> None:
    with pytest.raises(InvalidCommandDelayError):
        CommandDelayLine(delay_steps, NUM_ENVS, "cpu")  # type: ignore[arg-type]


def test_each_env_gets_back_its_own_commands() -> None:
    line = CommandDelayLine(1, NUM_ENVS, "cpu")
    commands = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    line.push(commands)
    delivered = line.push(torch.zeros(NUM_ENVS, 3))

    assert torch.equal(delivered, commands)
