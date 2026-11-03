"""指令の有効期限の判定と設定の検証。"""

import pytest

from cooperative_go2.entities.command_expiry import (
    CommandExpiryConfig,
    InvalidCommandExpiryConfigError,
    is_command_fresh,
    seconds_to_ns,
)

EXPIRY_NS = 100


@pytest.mark.parametrize(
    "command_stamp_ns,now_ns,expected",
    [(1000, 1000, True), (1000, 1100, True), (1000, 1101, False), (1050, 1000, True)],
    ids=["same_time", "at_expiry", "just_after_expiry", "newer_than_receiver"],
)
def test_freshness_boundary(command_stamp_ns: int, now_ns: int, expected: bool) -> None:
    assert is_command_fresh(command_stamp_ns, now_ns, EXPIRY_NS) is expected


def test_seconds_to_ns_rounds_control_period_exactly() -> None:
    assert seconds_to_ns(0.02) == 20_000_000
    assert seconds_to_ns(0.1) * 3 == seconds_to_ns(0.3)


@pytest.mark.parametrize(
    "velocity_s,joint_s", [(0.0, 0.1), (0.3, -0.1), (float("nan"), 0.1)], ids=["zero", "negative", "nan"]
)
def test_invalid_expiry_is_rejected(velocity_s: float, joint_s: float) -> None:
    with pytest.raises(InvalidCommandExpiryConfigError):
        CommandExpiryConfig(velocity_command_s=velocity_s, joint_target_s=joint_s)
