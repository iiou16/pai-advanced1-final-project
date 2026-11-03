import math

import pytest
import torch

from cooperative_go2.entities.push_config import PushRewardConfig, PushRewardScales
from cooperative_go2.entities.push_reward import (
    PUSH_REWARD_TERM_NAMES,
    PushRewardInputs,
    PushRewardTerms,
    compute_push_reward_terms,
    is_pushing_toward_goal,
)
from cooperative_go2.entities.push_termination import PushTerminationReason

SCALES = PushRewardScales(
    box_progress=10.0,
    success=5.0,
    intended_face_contact=0.5,
    box_rotation=-1.0,
    robot_fall=-2.0,
    unsafe_collision=-3.0,
    control_effort=-0.1,
)
CONFIG = PushRewardConfig(intended_face_min_cos=0.9, scales=SCALES)


def _inputs(**overrides: torch.Tensor) -> PushRewardInputs:
    """ロボットが箱の真後ろにいて何も起きていない 1 環境の材料を基準に、指定したフィールドだけ差し替える。"""
    values = {
        "previous_goal_distance_m": torch.tensor([2.0]),
        "goal_distance_m": torch.tensor([2.0]),
        "termination_reasons": torch.tensor([int(PushTerminationReason.NONE)]),
        "box_contact": torch.tensor([False]),
        "robot_xy_m": torch.tensor([[-1.0, 0.0]]),
        "box_xy_m": torch.tensor([[0.0, 0.0]]),
        "goal_xy_m": torch.tensor([[2.0, 0.0]]),
        "previous_box_yaw_rad": torch.tensor([0.0]),
        "box_yaw_rad": torch.tensor([0.0]),
        "fallen": torch.tensor([False]),
        "unsafe_collision": torch.tensor([False]),
        "commands": torch.zeros(1, 3),
    }
    return PushRewardInputs(**(values | overrides))


def test_term_names_match_scale_fields() -> None:
    assert set(PUSH_REWARD_TERM_NAMES) == set(SCALES.__dataclass_fields__)


def test_box_progress_is_decrease_in_goal_distance() -> None:
    inputs = _inputs(previous_goal_distance_m=torch.tensor([2.0, 2.0]), goal_distance_m=torch.tensor([1.75, 2.5]))

    terms = compute_push_reward_terms(inputs, CONFIG)

    # 近づけば正、遠ざかれば負
    assert torch.allclose(terms.box_progress, torch.tensor([0.25, -0.5]))


def test_robot_moving_alone_earns_no_progress_reward() -> None:
    # ロボットが目標へ大きく進んでも、箱と目標の距離が変わらなければ進捗の報酬は 0
    terms = compute_push_reward_terms(_inputs(robot_xy_m=torch.tensor([[1.5, 3.0]])), CONFIG)

    assert terms.box_progress.tolist() == [0.0]


def test_intended_face_contact_requires_contact_and_alignment() -> None:
    behind_without_contact = compute_push_reward_terms(_inputs(), CONFIG)
    behind_with_contact = compute_push_reward_terms(_inputs(box_contact=torch.tensor([True])), CONFIG)
    side_with_contact = compute_push_reward_terms(
        _inputs(box_contact=torch.tensor([True]), robot_xy_m=torch.tensor([[0.0, -1.0]])), CONFIG
    )

    assert behind_without_contact.intended_face_contact.tolist() == [0.0]
    assert behind_with_contact.intended_face_contact.tolist() == [1.0]
    assert side_with_contact.intended_face_contact.tolist() == [0.0]


def test_pushing_alignment_accepts_angles_up_to_threshold() -> None:
    # 1 環境目はしきい値ちょうどの角度、2 環境目はそれよりわずかに大きい角度でロボットが箱の後ろにいる。
    # しきい値ちょうどの余弦は丸め誤差で 0.9 をわずかに下回りうるため、下限をごくわずかに緩めて渡す
    angle = math.acos(0.9)
    robot = torch.tensor([[-math.cos(angle), -math.sin(angle)], [-math.cos(angle + 0.01), -math.sin(angle + 0.01)]])
    box = torch.zeros(2, 2)
    goal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    aligned = is_pushing_toward_goal(robot, box, goal, min_cos=0.9 - 1e-6)

    assert aligned.tolist() == [True, False]


def test_pushing_alignment_is_false_when_box_is_at_goal_center() -> None:
    aligned = is_pushing_toward_goal(torch.tensor([[-1.0, 0.0]]), torch.zeros(1, 2), torch.zeros(1, 2), min_cos=-1.0)

    assert aligned.tolist() == [False]


def test_box_rotation_uses_wrapped_yaw_change() -> None:
    inputs = _inputs(
        previous_box_yaw_rad=torch.tensor([math.pi - 0.1, 0.0]), box_yaw_rad=torch.tensor([-math.pi + 0.1, -0.3])
    )

    terms = compute_push_reward_terms(inputs, CONFIG)

    # ±π をまたいだ 0.2 rad の回転を、約 2π の回転と数えない
    assert torch.allclose(terms.box_rotation, torch.tensor([0.2, 0.3]), atol=1e-5)


def test_event_terms_are_one_when_event_happens() -> None:
    inputs = _inputs(fallen=torch.tensor([True]), unsafe_collision=torch.tensor([True]))

    terms = compute_push_reward_terms(inputs, CONFIG)

    assert (terms.robot_fall.item(), terms.unsafe_collision.item()) == (1.0, 1.0)


def test_success_reward_is_given_only_when_termination_reason_is_success() -> None:
    reasons = torch.tensor(
        [
            int(PushTerminationReason.SUCCESS),
            int(PushTerminationReason.FALL),
            int(PushTerminationReason.TIME_OUT),
            int(PushTerminationReason.NONE),
        ]
    )
    inputs = _inputs(
        termination_reasons=reasons,
        previous_goal_distance_m=torch.full((4,), 2.0),
        goal_distance_m=torch.full((4,), 2.0),
    )

    terms = compute_push_reward_terms(inputs, CONFIG)

    # 成功の条件を満たしていても、同じステップで転倒が優先された環境（2 番目）には成功の報酬を与えない
    assert terms.success.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_control_effort_is_sum_of_squared_command() -> None:
    terms = compute_push_reward_terms(_inputs(commands=torch.tensor([[0.3, -0.4, 1.0]])), CONFIG)

    assert torch.allclose(terms.control_effort, torch.tensor([1.25]))


def test_weighted_terms_multiply_each_scale_and_total_sums_them() -> None:
    terms = PushRewardTerms(**{name: torch.tensor([1.0]) for name in PUSH_REWARD_TERM_NAMES})

    weighted = terms.weighted(SCALES)

    assert {name: value.item() for name, value in weighted.named().items()} == pytest.approx(
        {name: getattr(SCALES, name) for name in PUSH_REWARD_TERM_NAMES}
    )
    assert weighted.total().item() == pytest.approx(10.0 + 5.0 + 0.5 - 1.0 - 2.0 - 3.0 - 0.1)
