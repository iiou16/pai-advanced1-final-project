"""シミュレーションノードが適用する関節目標の選び方（学習時と同じ 1 周期遅れ・有効期限・エピソードの区切り）。"""

import pytest
import torch

from cooperative_go2.use_cases.joint_target_scheduler import (
    InvalidJointTargetTimingError,
    InvalidSelectedJointTargetsError,
    JointTargetScheduler,
    JointTargetTiming,
    SelectedJointTargets,
    TargetSelection,
)

DT_NS = 20
START_NS = 1000
DEFAULT_TARGETS = torch.zeros(1, 12)


def _targets(value: float) -> torch.Tensor:
    return torch.full((1, 12), value)


@pytest.fixture
def scheduler() -> JointTargetScheduler:
    timing = JointTargetTiming(control_dt_ns=DT_NS, latency_steps=1, expiry_ns=5 * DT_NS)
    scheduler = JointTargetScheduler(timing, DEFAULT_TARGETS)
    scheduler.start_episode(START_NS)
    return scheduler


def test_first_step_of_episode_holds_default_pose_like_training(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))

    selected = scheduler.select(START_NS)

    assert selected.selection is TargetSelection.EPISODE_START
    assert torch.equal(selected.joint_targets, DEFAULT_TARGETS)


def test_target_from_previous_state_is_applied_even_if_newer_one_arrived(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))
    scheduler.receive(START_NS + DT_NS, _targets(2.0))

    selected = scheduler.select(START_NS + DT_NS)

    assert selected.selection is TargetSelection.POLICY
    assert torch.equal(selected.joint_targets, _targets(1.0))
    assert selected.delay_steps == 1


def test_older_target_is_used_when_latest_has_not_arrived(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))

    selected = scheduler.select(START_NS + 3 * DT_NS)

    assert selected.selection is TargetSelection.POLICY
    assert selected.delay_steps == 3


def test_target_older_than_expiry_is_replaced_by_default_pose(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))

    selected = scheduler.select(START_NS + 6 * DT_NS)

    assert selected.selection is TargetSelection.STALE
    assert torch.equal(selected.joint_targets, DEFAULT_TARGETS)


def test_no_target_after_episode_start_period_is_stale(scheduler: JointTargetScheduler) -> None:
    selected = scheduler.select(START_NS + DT_NS)

    assert selected.selection is TargetSelection.STALE


def test_targets_from_previous_episode_are_discarded(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))
    next_start = START_NS + 10 * DT_NS
    scheduler.start_episode(next_start)
    scheduler.receive(START_NS + 5 * DT_NS, _targets(9.0))

    selected = scheduler.select(next_start + DT_NS)

    assert selected.selection is TargetSelection.STALE


def test_target_at_exact_expiry_is_still_applied(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))

    at_expiry = scheduler.select(START_NS + 5 * DT_NS)

    assert at_expiry.selection is TargetSelection.POLICY
    assert at_expiry.delay_steps == 5


def test_reused_target_grows_older_until_stale(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS, _targets(1.0))

    selections = [scheduler.select(START_NS + steps * DT_NS) for steps in range(1, 8)]

    assert [selected.delay_steps for selected in selections[:5]] == [1, 2, 3, 4, 5]
    assert all(selected.selection is TargetSelection.STALE for selected in selections[5:])


def test_without_latency_target_of_current_state_is_applied_immediately() -> None:
    timing = JointTargetTiming(control_dt_ns=DT_NS, latency_steps=0, expiry_ns=5 * DT_NS)
    scheduler = JointTargetScheduler(timing, DEFAULT_TARGETS)
    scheduler.start_episode(START_NS)
    scheduler.receive(START_NS, _targets(1.0))

    selected = scheduler.select(START_NS)

    assert selected.selection is TargetSelection.POLICY
    assert selected.delay_steps == 0


def test_without_latency_missing_target_at_start_is_stale() -> None:
    timing = JointTargetTiming(control_dt_ns=DT_NS, latency_steps=0, expiry_ns=5 * DT_NS)
    scheduler = JointTargetScheduler(timing, DEFAULT_TARGETS)
    scheduler.start_episode(START_NS)

    assert scheduler.select(START_NS).selection is TargetSelection.STALE


def test_target_stamped_at_episode_start_is_accepted_but_earlier_is_dropped(scheduler: JointTargetScheduler) -> None:
    scheduler.receive(START_NS - 1, _targets(9.0))
    scheduler.receive(START_NS, _targets(1.0))

    selected = scheduler.select(START_NS + DT_NS)

    assert torch.equal(selected.joint_targets, _targets(1.0))


def test_default_pose_is_copied_at_construction() -> None:
    default = torch.zeros(1, 12)
    scheduler = JointTargetScheduler(JointTargetTiming(DT_NS, latency_steps=1, expiry_ns=5 * DT_NS), default)
    scheduler.start_episode(START_NS)
    default.fill_(3.0)

    assert torch.count_nonzero(scheduler.select(START_NS).joint_targets) == 0


@pytest.mark.parametrize(
    "control_dt_ns,latency_steps,expiry_ns",
    [(0, 1, 100), (DT_NS, -1, 5 * DT_NS), (DT_NS, 1, DT_NS)],
    ids=["zero_dt", "negative_latency", "expiry_shorter_than_latency_plus_one_period"],
)
def test_invalid_timing_is_rejected(control_dt_ns: int, latency_steps: int, expiry_ns: int) -> None:
    with pytest.raises(InvalidJointTargetTimingError):
        JointTargetTiming(control_dt_ns=control_dt_ns, latency_steps=latency_steps, expiry_ns=expiry_ns)


@pytest.mark.parametrize(
    "selection,delay_steps",
    [(TargetSelection.POLICY, None), (TargetSelection.STALE, 2), (TargetSelection.EPISODE_START, 0)],
    ids=["policy_without_delay", "stale_with_delay", "episode_start_with_delay"],
)
def test_selection_and_delay_must_agree(selection: TargetSelection, delay_steps: int | None) -> None:
    with pytest.raises(InvalidSelectedJointTargetsError):
        SelectedJointTargets(DEFAULT_TARGETS, selection, delay_steps)
