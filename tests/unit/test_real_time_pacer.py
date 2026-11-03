"""物理ステップの実時間の間隔の調整の検証。"""

import pytest

from cooperative_go2.interface_adapters.real_time_pacer import RealTimePacer


class FakeWallClock:
    """sleep で進む実時間。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


@pytest.fixture
def clock() -> FakeWallClock:
    return FakeWallClock()


def test_waits_until_min_interval_after_previous_step(clock: FakeWallClock) -> None:
    pacer = RealTimePacer(control_dt_s=0.02, max_real_time_factor=2.0, wall_clock=clock, sleep=clock.sleep)

    pacer.wait_for_next_step()
    clock.now += 0.004
    pacer.wait_for_next_step()

    assert clock.sleeps == [pytest.approx(0.006)]


def test_does_not_wait_when_computation_is_slower_than_interval(clock: FakeWallClock) -> None:
    pacer = RealTimePacer(control_dt_s=0.02, max_real_time_factor=1.0, wall_clock=clock, sleep=clock.sleep)

    pacer.wait_for_next_step()
    clock.now += 0.05
    pacer.wait_for_next_step()

    assert clock.sleeps == []
