import pytest
import torch

from cooperative_go2.entities.episode_reward_tracker import EpisodeRewardTracker
from cooperative_go2.entities.gait_reward import GAIT_REWARD_TERM_NAMES, GaitRewardTerms
from cooperative_go2.entities.gait_termination import TerminationReason

CONTROL_DT_S = 0.5


def _terms(values: list[float], **overrides: list[float]) -> GaitRewardTerms:
    """すべての項を values にし、overrides で指定した項だけ別の値にした報酬の項。"""
    return GaitRewardTerms(
        **{name: torch.tensor(overrides.get(name, values)) for name in GAIT_REWARD_TERM_NAMES}
    )


def _reasons(*reasons: TerminationReason) -> torch.Tensor:
    return torch.tensor([int(reason) for reason in reasons])


def test_no_finished_episode_gives_none() -> None:
    tracker = EpisodeRewardTracker(num_envs=2, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([1.0, 1.0]))

    summary = tracker.summarize_completed_or_none(
        torch.tensor([False, False]), _reasons(TerminationReason.NONE, TerminationReason.NONE)
    )

    assert summary is None


def test_reward_per_second_uses_each_episode_actual_length() -> None:
    tracker = EpisodeRewardTracker(num_envs=2, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([1.0, 1.0]))
    tracker.reset_where(torch.tensor([True, False]))
    tracker.add_step(_terms([3.0, 1.0]))

    summary = tracker.summarize_completed_or_none(
        torch.tensor([True, True]), _reasons(TerminationReason.TIME_OUT, TerminationReason.TIME_OUT)
    )

    # 環境 0: リセット後の 1 ステップ（0.5 s）で報酬 3 → 6/s。環境 1: 2 ステップ（1 s）で報酬 2 → 2/s
    assert summary is not None
    assert summary.reward_per_second[GAIT_REWARD_TERM_NAMES[0]].item() == pytest.approx(4.0)


def test_each_term_is_summarized_separately() -> None:
    tracker = EpisodeRewardTracker(num_envs=1, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([0.0], tracking_ang_vel=[1.0], action_rate=[-2.0]))

    summary = tracker.summarize_completed_or_none(torch.tensor([True]), _reasons(TerminationReason.FALL))

    assert summary is not None
    log = summary.as_log()
    assert log["rew_tracking_ang_vel"].item() == pytest.approx(2.0)
    assert log["rew_action_rate"].item() == pytest.approx(-4.0)
    assert log["rew_tracking_lin_vel"].item() == pytest.approx(0.0)


def test_summarizing_does_not_clear_the_sums() -> None:
    tracker = EpisodeRewardTracker(num_envs=1, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([1.0]))
    done = torch.tensor([True])
    reasons = _reasons(TerminationReason.TIME_OUT)

    first = tracker.summarize_completed_or_none(done, reasons)
    second = tracker.summarize_completed_or_none(done, reasons)

    assert first is not None and second is not None
    assert torch.equal(first.reward_per_second["tracking_lin_vel"], second.reward_per_second["tracking_lin_vel"])


def test_reset_discards_partial_episode_of_masked_envs_only() -> None:
    tracker = EpisodeRewardTracker(num_envs=2, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([10.0, 10.0]))
    tracker.reset_where(torch.tensor([True, False]))
    tracker.add_step(_terms([1.0, 1.0]))

    summary = tracker.summarize_completed_or_none(
        torch.tensor([True, False]), _reasons(TerminationReason.TIME_OUT, TerminationReason.NONE)
    )

    # 環境 0 はリセット前の 10 を含まない: 1 ステップ（0.5 s）で 1 → 2/s
    assert summary is not None
    assert summary.reward_per_second["tracking_lin_vel"].item() == pytest.approx(2.0)


def test_termination_fraction_counts_only_finished_envs() -> None:
    tracker = EpisodeRewardTracker(num_envs=4, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([0.0] * 4))
    reasons = _reasons(
        TerminationReason.FALL, TerminationReason.TIME_OUT, TerminationReason.FALL, TerminationReason.NONE
    )

    summary = tracker.summarize_completed_or_none(torch.tensor([True, True, True, False]), reasons)

    assert summary is not None
    assert summary.termination_fraction[TerminationReason.FALL].item() == pytest.approx(2 / 3)
    assert summary.termination_fraction[TerminationReason.TIME_OUT].item() == pytest.approx(1 / 3)
    assert summary.termination_fraction[TerminationReason.SIM_DIVERGED].item() == pytest.approx(0.0)


def test_log_keys_are_prefixed_by_kind() -> None:
    tracker = EpisodeRewardTracker(num_envs=1, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([0.0]))

    summary = tracker.summarize_completed_or_none(torch.tensor([True]), _reasons(TerminationReason.TIME_OUT))

    assert summary is not None
    assert set(summary.as_log()) == {f"rew_{name}" for name in GAIT_REWARD_TERM_NAMES} | {
        "termination_time_out", "termination_fall", "termination_sim_diverged",
    }  # fmt: skip
