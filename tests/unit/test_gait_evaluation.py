import pytest
import torch

from cooperative_go2.entities.gait_evaluation import GaitTrackingAccumulator
from cooperative_go2.entities.velocity_command import VelocityCommand

FORWARD = VelocityCommand(vx_mps=0.5, vy_mps=0.0, yaw_rate_radps=0.0)


def test_tracking_errors_are_averaged_over_envs_and_steps() -> None:
    accumulator = GaitTrackingAccumulator(FORWARD, num_envs=2, device="cpu")
    no_fall = torch.tensor([False, False])

    # 環境 0 は指令どおり、環境 1 は静止して 0.5 m/s 遅れ、ヨーが 0.2 rad/s ずれる
    accumulator.add_step(
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]),
        fallen=no_fall,
    )
    summary = accumulator.summarize()

    assert summary.mean_lin_vel_error_mps == pytest.approx(0.25)
    assert summary.mean_yaw_rate_error_radps == pytest.approx(0.1)
    assert summary.num_steps == 1


def test_fall_rate_counts_envs_that_fell_at_least_once() -> None:
    accumulator = GaitTrackingAccumulator(FORWARD, num_envs=4, device="cpu")
    velocity = torch.zeros(4, 3)

    accumulator.add_step(velocity, velocity, fallen=torch.tensor([True, False, False, False]))
    accumulator.add_step(velocity, velocity, fallen=torch.tensor([True, True, False, False]))
    summary = accumulator.summarize()

    assert summary.num_falls == 3
    assert summary.fall_rate == pytest.approx(0.5)


def test_summarize_without_steps_raises() -> None:
    accumulator = GaitTrackingAccumulator(FORWARD, num_envs=1, device="cpu")

    with pytest.raises(ValueError):
        accumulator.summarize()
