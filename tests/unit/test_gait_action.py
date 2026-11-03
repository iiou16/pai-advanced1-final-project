import pytest
import torch

from cooperative_go2.entities.gait_action import GaitActionHistory, to_joint_targets


def test_advance_clips_new_action_and_shifts_previous() -> None:
    history = GaitActionHistory.zeros(num_envs=1, num_joints=2, device="cpu")

    first = history.advance(torch.tensor([[0.5, 150.0]]), clip_limit=100.0)
    second = first.advance(torch.tensor([[-200.0, 1.0]]), clip_limit=100.0)

    assert torch.equal(first.current, torch.tensor([[0.5, 100.0]]))
    assert torch.equal(second.current, torch.tensor([[-100.0, 1.0]]))
    assert torch.equal(second.previous, first.current)


@pytest.mark.parametrize(
    ("simulate_latency", "expected"), [(True, [[1.0]]), (False, [[2.0]])], ids=["latency", "no-latency"]
)
def test_applied_action_depends_on_latency(simulate_latency: bool, expected: list[list[float]]) -> None:
    history = GaitActionHistory(current=torch.tensor([[2.0]]), previous=torch.tensor([[1.0]]))

    assert torch.equal(history.applied(simulate_latency), torch.tensor(expected))


def test_reset_clears_both_current_and_previous_of_masked_envs() -> None:
    history = GaitActionHistory(current=torch.ones(2, 3), previous=torch.full((2, 3), 2.0))

    reset = history.reset_where(torch.tensor([True, False]))

    assert torch.equal(reset.current[0], torch.zeros(3))
    assert torch.equal(reset.previous[0], torch.zeros(3))
    assert torch.equal(reset.current[1], torch.ones(3))
    assert torch.equal(reset.previous[1], torch.full((3,), 2.0))


def test_step_sequence_matches_latency_semantics() -> None:
    # 1 周期遅延: 方策の行動 a1, a2 に対し、適用されるのは 0, a1。リセット後は再び 0 から始まる
    history = GaitActionHistory.zeros(num_envs=1, num_joints=1, device="cpu")
    applied = []
    for raw in (1.0, 2.0):
        history = history.advance(torch.tensor([[raw]]), clip_limit=100.0)
        applied.append(history.applied(simulate_latency=True).item())
    history = history.reset_where(torch.tensor([True])).advance(torch.tensor([[3.0]]), clip_limit=100.0)
    applied.append(history.applied(simulate_latency=True).item())

    assert applied == [0.0, 1.0, 0.0]


def test_joint_targets_are_default_pose_plus_scaled_action() -> None:
    targets = to_joint_targets(torch.tensor([[1.0, -2.0]]), torch.tensor([0.8, -1.5]), action_scale=0.25)

    assert torch.allclose(targets, torch.tensor([[1.05, -2.0]]))
