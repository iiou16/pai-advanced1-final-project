import pytest
import torch

from cooperative_go2.entities.gait_evaluation import (
    CommandSegment,
    GaitEvaluationScenario,
    GaitEvaluationSuite,
    InvalidEvaluationScenarioError,
    SegmentTrackingAccumulator,
)
from cooperative_go2.entities.velocity_command import VelocityCommand

FORWARD = VelocityCommand(vx_mps=0.5, vy_mps=0.0, yaw_rate_radps=0.0)
FORWARD_SEGMENT = CommandSegment(command=FORWARD, duration_s=2.0)


def test_tracking_errors_are_averaged_over_envs_and_steps() -> None:
    accumulator = SegmentTrackingAccumulator(FORWARD_SEGMENT, num_envs=2, device="cpu")

    # 環境 0 は指令どおり、環境 1 は静止して 0.5 m/s 遅れ、ヨーが 0.2 rad/s ずれる
    accumulator.add_tracking(
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]),
    )
    summary = accumulator.summarize()

    assert summary.mean_lin_vel_error_mps == pytest.approx(0.25)
    assert summary.mean_yaw_rate_error_radps == pytest.approx(0.1)
    assert summary.num_tracking_steps == 1
    assert summary.duration_s == pytest.approx(2.0)


def test_fall_rate_counts_envs_that_fell_at_least_once() -> None:
    accumulator = SegmentTrackingAccumulator(FORWARD_SEGMENT, num_envs=4, device="cpu")
    accumulator.add_tracking(torch.zeros(4, 3), torch.zeros(4, 3))

    accumulator.add_falls(torch.tensor([True, False, False, False]))
    accumulator.add_falls(torch.tensor([True, True, False, False]))
    summary = accumulator.summarize()

    assert summary.num_falls == 3
    assert summary.fall_rate == pytest.approx(0.5)


def test_summarize_without_tracking_steps_raises() -> None:
    accumulator = SegmentTrackingAccumulator(FORWARD_SEGMENT, num_envs=1, device="cpu")
    accumulator.add_falls(torch.tensor([False]))

    with pytest.raises(ValueError):
        accumulator.summarize()


@pytest.mark.parametrize(
    ("command", "duration_s"),
    [
        (VelocityCommand(vx_mps=float("nan"), vy_mps=0.0, yaw_rate_radps=0.0), 1.0),
        (FORWARD, 0.0),
        (FORWARD, float("inf")),
    ],
    ids=["nan-command", "zero-duration", "inf-duration"],
)
def test_invalid_segment_is_rejected(command: VelocityCommand, duration_s: float) -> None:
    with pytest.raises(InvalidEvaluationScenarioError):
        CommandSegment(command=command, duration_s=duration_s)


@pytest.mark.parametrize("name", ["", "with space", "../escape", "a/b"], ids=["empty", "space", "parent", "slash"])
def test_scenario_name_must_be_usable_as_file_name(name: str) -> None:
    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationScenario(name=name, segments=(FORWARD_SEGMENT,))


def test_scenario_without_segments_is_rejected() -> None:
    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationScenario(name="empty", segments=())


def test_scenario_duration_is_sum_of_segments() -> None:
    scenario = GaitEvaluationScenario(name="twice", segments=(FORWARD_SEGMENT, FORWARD_SEGMENT))

    assert scenario.duration_s() == pytest.approx(4.0)


def test_suite_rejects_duplicate_scenario_names() -> None:
    scenario = GaitEvaluationScenario(name="forward", segments=(FORWARD_SEGMENT,))

    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationSuite(settle_time_s=0.5, scenarios=(scenario, scenario))


def test_suite_rejects_settle_time_not_shorter_than_a_segment() -> None:
    scenario = GaitEvaluationScenario(name="forward", segments=(FORWARD_SEGMENT,))

    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationSuite(settle_time_s=FORWARD_SEGMENT.duration_s, scenarios=(scenario,))


@pytest.mark.parametrize("settle_time_s", [-0.1, float("nan")], ids=["negative", "nan"])
def test_suite_rejects_invalid_settle_time(settle_time_s: float) -> None:
    scenario = GaitEvaluationScenario(name="forward", segments=(FORWARD_SEGMENT,))

    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationSuite(settle_time_s=settle_time_s, scenarios=(scenario,))


def test_suite_without_scenarios_is_rejected() -> None:
    with pytest.raises(InvalidEvaluationScenarioError):
        GaitEvaluationSuite(settle_time_s=0.0, scenarios=())
