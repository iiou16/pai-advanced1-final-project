import math

import pytest

from cooperative_go2.entities.config_validation import (
    ConfigValueChecker,
    InvalidConfigError,
    ceil_ratio,
    is_integer_ratio,
)


class _SampleConfigError(InvalidConfigError):
    """検査が指定したドメイン例外で違反を知らせることを確かめるための例外。"""


CHECK = ConfigValueChecker(_SampleConfigError)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("check_name", ["finite", "positive", "non_negative", "unit_interval"])
def test_every_check_rejects_non_finite_value_with_domain_error(check_name: str, value: float) -> None:
    with pytest.raises(_SampleConfigError, match="section.key は有限の値が必要"):
        getattr(CHECK, check_name)("section.key", value)


@pytest.mark.parametrize(
    ("check_name", "rejected"), [("positive", 0.0), ("non_negative", -1e-9), ("unit_interval", 1.01)]
)
def test_check_rejects_value_just_outside_its_range(check_name: str, rejected: float) -> None:
    with pytest.raises(_SampleConfigError, match="section.key"):
        getattr(CHECK, check_name)("section.key", rejected)


@pytest.mark.parametrize(
    ("check_name", "accepted"), [("positive", 1e-9), ("non_negative", 0.0), ("unit_interval", 1.0)]
)
def test_check_accepts_value_at_edge_of_its_range(check_name: str, accepted: float) -> None:
    getattr(CHECK, check_name)("section.key", accepted)


def test_domain_error_is_catchable_as_common_base_error() -> None:
    with pytest.raises(InvalidConfigError):
        CHECK.positive("section.key", -1.0)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(4.0, 0.02, True), (0.3, 0.1, True), (0.25, 0.1, False)],
)
def test_integer_ratio_tolerates_rounding_error(numerator: float, denominator: float, expected: bool) -> None:
    assert is_integer_ratio(numerator, denominator) is expected


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(0.3, 0.1, 3), (20.0, 0.02, 1000), (0.25, 0.1, 3), (0.01, 0.1, 1)],
)
def test_ceil_ratio_rounds_up_partial_step_but_not_rounding_error(
    numerator: float, denominator: float, expected: int
) -> None:
    assert ceil_ratio(numerator, denominator) == expected
