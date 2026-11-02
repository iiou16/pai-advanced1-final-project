import pytest
import torch

from cooperative_go2.entities.episode_reward_tracker import EpisodeRewardTracker
from cooperative_go2.entities.gait_reward import GAIT_REWARD_TERM_NAMES
from cooperative_go2.entities.gait_termination import TerminationReason

CONTROL_DT_S = 0.5


def _terms(value: list[float]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(value) for name in GAIT_REWARD_TERM_NAMES}


def test_no_finished_episode_gives_none() -> None:
    tracker = EpisodeRewardTracker(num_envs=2, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([1.0, 1.0]))

    summary = tracker.pop_completed_or_none(torch.tensor([False, False]), torch.zeros(2, dtype=torch.int64))

    assert summary is None


def test_reward_per_second_uses_each_episode_actual_length() -> None:
    tracker = EpisodeRewardTracker(num_envs=2, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([1.0, 1.0]))
    reasons = torch.tensor([int(TerminationReason.FALL), int(TerminationReason.NONE)])
    tracker.pop_completed_or_none(torch.tensor([True, False]), reasons)
    tracker.add_step(_terms([3.0, 1.0]))

    summary = tracker.pop_completed_or_none(
        torch.tensor([True, True]),
        torch.tensor([int(TerminationReason.TIME_OUT), int(TerminationReason.TIME_OUT)]),
    )

    # 環境 0: 2 回目のエピソードは 1 ステップ（0.5 s）で報酬 3 → 6/s。環境 1: 2 ステップ（1 s）で報酬 2 → 2/s
    assert summary is not None
    assert summary.reward_per_second[GAIT_REWARD_TERM_NAMES[0]].item() == pytest.approx(4.0)


def test_termination_fraction_counts_only_finished_envs() -> None:
    tracker = EpisodeRewardTracker(num_envs=4, control_dt_s=CONTROL_DT_S, device="cpu")
    tracker.add_step(_terms([0.0] * 4))
    reasons = torch.tensor([int(reason) for reason in (
        TerminationReason.FALL, TerminationReason.TIME_OUT, TerminationReason.FALL, TerminationReason.NONE,
    )])  # fmt: skip

    summary = tracker.pop_completed_or_none(torch.tensor([True, True, True, False]), reasons)

    assert summary is not None
    assert summary.termination_fraction[TerminationReason.FALL].item() == pytest.approx(2 / 3)
    assert summary.termination_fraction[TerminationReason.TIME_OUT].item() == pytest.approx(1 / 3)
    assert summary.termination_fraction[TerminationReason.SIM_DIVERGED].item() == pytest.approx(0.0)
    assert set(summary.as_log()) == {f"rew_{name}" for name in GAIT_REWARD_TERM_NAMES} | {
        "termination_time_out", "termination_fall", "termination_sim_diverged",
    }  # fmt: skip


def test_missing_reward_term_raises_key_error() -> None:
    tracker = EpisodeRewardTracker(num_envs=1, control_dt_s=CONTROL_DT_S, device="cpu")

    with pytest.raises(KeyError):
        tracker.add_step({GAIT_REWARD_TERM_NAMES[0]: torch.zeros(1)})
