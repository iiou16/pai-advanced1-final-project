import torch

from cooperative_go2.entities.gait_termination import TerminationReason, classify_terminations, exceeds_tilt_limit


def test_roll_and_pitch_are_checked_against_their_own_limits() -> None:
    euler_deg = torch.tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, -31.0, 0.0], [5.0, 5.0, 170.0]])

    fallen = exceeds_tilt_limit(euler_deg, roll_limit_deg=10.0, pitch_limit_deg=30.0)

    # ロール 20 は上限 10 を超え、ピッチ 20 は上限 30 以内。ヨーはいくら回っても転倒ではない
    assert fallen.tolist() == [True, False, True, False]


def test_tilt_exactly_at_limit_is_not_a_fall() -> None:
    fallen = exceeds_tilt_limit(torch.tensor([[10.0, -10.0, 0.0]]), roll_limit_deg=10.0, pitch_limit_deg=10.0)

    assert fallen.tolist() == [False]


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
