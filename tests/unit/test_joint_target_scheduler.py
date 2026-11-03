"""シミュレーションノードが適用する関節目標の選び方（学習時と同じ 1 周期遅れ・有効期限・エピソードの区切り）。"""

import pytest
import torch

from cooperative_go2.use_cases.joint_target_scheduler import (
    JointTargetScheduler,
    JointTargetTiming,
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
