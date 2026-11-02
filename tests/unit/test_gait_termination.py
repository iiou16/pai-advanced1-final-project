import torch

from cooperative_go2.entities.gait_termination import TerminationReason, classify_terminations, exceeds_tilt_limit


def test_tilt_beyond_roll_or_pitch_limit_counts_as_fall() -> None:
    euler_deg = torch.tensor([[5.0, 5.0, 90.0], [-11.0, 0.0, 0.0], [0.0, 10.5, 0.0]])

    fallen = exceeds_tilt_limit(euler_deg, roll_limit_deg=10.0, pitch_limit_deg=10.0)

    # ヨーはいくら回っても転倒ではない
    assert fallen.tolist() == [False, True, True]


def test_reason_priority_is_diverged_then_fall_then_time_out() -> None:
    time_out = torch.tensor([False, True, True, True])
    fallen = torch.tensor([False, False, True, True])
    diverged = torch.tensor([False, False, False, True])

    reasons = classify_terminations(time_out, fallen, diverged)

    assert reasons.tolist() == [
        TerminationReason.NONE,
        TerminationReason.TIME_OUT,
        TerminationReason.FALL,
        TerminationReason.SIM_DIVERGED,
    ]
