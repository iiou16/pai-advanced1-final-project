import pytest
import torch

from cooperative_go2.entities.partner_channel import DelayedPartnerChannel, PartnerMessage
from cooperative_go2.entities.team_marl_config import InvalidTeamMarlConfigError, TeamCommunicationConfig

NUM_ENVS = 2
PERIOD_S = 0.1
MAX_AGE_S = 1.0
CPU = torch.device("cpu")


def _config(drop_probability: float = 0.0) -> TeamCommunicationConfig:
    return TeamCommunicationConfig(delay_s=PERIOD_S, drop_probability=drop_probability, max_age_s=MAX_AGE_S)


def _message(x_m: float, stamp_s: float) -> PartnerMessage:
    return PartnerMessage(
        position_xy_m=torch.tensor([[x_m, 0.0]] * NUM_ENVS),
        yaw_rad=torch.zeros(NUM_ENVS),
        command=torch.full((NUM_ENVS, 3), x_m),
        stamp_s=torch.full((NUM_ENVS,), stamp_s),
    )


def _now(time_s: float) -> torch.Tensor:
    return torch.full((NUM_ENVS,), time_s)


def test_nothing_is_received_before_the_first_transmission() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=1, num_envs=NUM_ENVS, device=CPU)

    reception = channel.reception(_now(0.0))

    assert not reception.received.any()
    assert torch.equal(reception.age_s, torch.full((NUM_ENVS,), MAX_AGE_S))
    assert torch.equal(reception.message.position_xy_m, torch.zeros(NUM_ENVS, 2))


def test_one_step_delay_delivers_the_message_sent_in_the_previous_step() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=1, num_envs=NUM_ENVS, device=CPU)

    channel.transmit(_message(x_m=1.0, stamp_s=0.0), torch.Generator())
    reception = channel.reception(_now(PERIOD_S))

    assert reception.received.all()
    assert torch.allclose(reception.message.position_xy_m[:, 0], torch.ones(NUM_ENVS))
    assert torch.allclose(reception.age_s, torch.full((NUM_ENVS,), PERIOD_S))


def test_two_step_delay_holds_the_message_for_one_more_step() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=2, num_envs=NUM_ENVS, device=CPU)

    channel.transmit(_message(x_m=1.0, stamp_s=0.0), torch.Generator())
    first = channel.reception(_now(PERIOD_S))
    channel.transmit(_message(x_m=2.0, stamp_s=PERIOD_S), torch.Generator())
    second = channel.reception(_now(2 * PERIOD_S))

    assert not first.received.any()
    assert torch.allclose(second.message.position_xy_m[:, 0], torch.ones(NUM_ENVS))
    assert torch.allclose(second.age_s, torch.full((NUM_ENVS,), 2 * PERIOD_S))


def test_dropped_messages_keep_the_last_received_message_and_its_age_grows() -> None:
    num_envs = 200
    channel = DelayedPartnerChannel(_config(drop_probability=0.5), delay_steps=1, num_envs=num_envs, device=CPU)
    generator = torch.Generator().manual_seed(0)

    def message(x_m: float, stamp_s: float) -> PartnerMessage:
        return PartnerMessage(
            position_xy_m=torch.tensor([[x_m, 0.0]] * num_envs), yaw_rad=torch.zeros(num_envs),
            command=torch.zeros(num_envs, 3), stamp_s=torch.full((num_envs,), stamp_s),
        )  # fmt: skip

    channel.transmit(message(x_m=1.0, stamp_s=0.0), generator)
    channel.transmit(message(x_m=2.0, stamp_s=PERIOD_S), generator)
    reception = channel.reception(torch.full((num_envs,), 2 * PERIOD_S))

    received_x = reception.message.position_xy_m[reception.received, 0]
    received_age = reception.age_s[reception.received]
    kept_old = received_x == 1.0
    # 2 回目が落ちた環境は 1 回目の状態を使い続け、その古さは 2 周期になる
    assert kept_old.any() and (received_x == 2.0).any()
    assert torch.allclose(received_age[kept_old], torch.full((int(kept_old.sum()),), 2 * PERIOD_S))
    assert torch.allclose(received_age[~kept_old], torch.full((int((~kept_old).sum()),), PERIOD_S))


def test_drop_rate_matches_the_configured_probability() -> None:
    num_envs = 20000
    channel = DelayedPartnerChannel(_config(drop_probability=0.3), delay_steps=1, num_envs=num_envs, device=CPU)
    message = PartnerMessage(
        position_xy_m=torch.zeros(num_envs, 2), yaw_rad=torch.zeros(num_envs),
        command=torch.zeros(num_envs, 3), stamp_s=torch.zeros(num_envs),
    )  # fmt: skip

    channel.transmit(message, torch.Generator().manual_seed(1))

    received_fraction = channel.reception(torch.full((num_envs,), PERIOD_S)).received.float().mean()
    assert received_fraction == pytest.approx(0.7, abs=0.02)


def test_age_is_capped_at_max_age() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=1, num_envs=NUM_ENVS, device=CPU)
    channel.transmit(_message(x_m=1.0, stamp_s=0.0), torch.Generator())

    reception = channel.reception(_now(5.0))

    assert torch.equal(reception.age_s, torch.full((NUM_ENVS,), MAX_AGE_S))


def test_reset_discards_received_and_in_flight_messages_only_for_reset_envs() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=2, num_envs=NUM_ENVS, device=CPU)
    channel.transmit(_message(x_m=1.0, stamp_s=0.0), torch.Generator())
    channel.transmit(_message(x_m=2.0, stamp_s=PERIOD_S), torch.Generator())

    channel.reset_where(torch.tensor([True, False]))
    after_reset = channel.reception(_now(2 * PERIOD_S))
    channel.transmit(_message(x_m=3.0, stamp_s=0.0), torch.Generator())
    next_step = channel.reception(_now(3 * PERIOD_S))

    assert after_reset.received.tolist() == [False, True]
    # 環境 0 では、リセット前に送られていた状態（x = 2）を届けない
    assert next_step.received.tolist() == [False, True]
    assert next_step.message.position_xy_m[1, 0] == pytest.approx(2.0)


def test_zero_drop_probability_does_not_consume_random_numbers() -> None:
    channel = DelayedPartnerChannel(_config(), delay_steps=1, num_envs=NUM_ENVS, device=CPU)
    generator = torch.Generator().manual_seed(3)
    expected = torch.rand(1, generator=torch.Generator().manual_seed(3))

    channel.transmit(_message(x_m=1.0, stamp_s=0.0), generator)

    assert torch.equal(torch.rand(1, generator=generator), expected)


def test_delay_shorter_than_one_period_is_rejected() -> None:
    config = TeamCommunicationConfig(delay_s=0.05, drop_probability=0.0, max_age_s=MAX_AGE_S)

    with pytest.raises(InvalidTeamMarlConfigError):
        config.delay_steps(PERIOD_S)


def test_drop_probability_of_one_is_rejected() -> None:
    with pytest.raises(InvalidTeamMarlConfigError):
        TeamCommunicationConfig(delay_s=PERIOD_S, drop_probability=1.0, max_age_s=MAX_AGE_S)
