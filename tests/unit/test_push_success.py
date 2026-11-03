import torch

from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region


def test_goal_region_includes_its_boundary() -> None:
    goal = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    box = torch.tensor([[1.0, 1.0], [1.5, 1.0], [1.0, 1.5001]])

    in_goal = is_in_goal_region(box, goal, goal_radius_m=0.5)

    # 中心・ちょうど境界上は内側、境界をわずかに超えたら外側
    assert in_goal.tolist() == [True, True, False]


def test_hold_steps_count_up_while_inside_goal() -> None:
    hold_steps = torch.tensor([0, 3])

    advanced = advance_goal_hold_steps(hold_steps, torch.tensor([True, True]))

    assert advanced.tolist() == [1, 4]


def test_hold_steps_reset_when_box_leaves_goal() -> None:
    hold_steps = torch.tensor([5, 0])

    advanced = advance_goal_hold_steps(hold_steps, torch.tensor([False, False]))

    assert advanced.tolist() == [0, 0]


def test_hold_steps_do_not_change_input_tensor() -> None:
    hold_steps = torch.tensor([2])

    advance_goal_hold_steps(hold_steps, torch.tensor([True]))

    assert hold_steps.tolist() == [2]


def test_success_requires_reaching_required_steps() -> None:
    hold_steps = torch.tensor([0, 4, 5, 6])

    succeeded = has_held_goal(hold_steps, required_hold_steps=5)

    # 1 ステップ足りなければ未成功、ちょうど達した時点で成功
    assert succeeded.tolist() == [False, False, True, True]


def test_leaving_goal_one_step_before_success_restarts_the_count() -> None:
    required = 3
    hold_steps = torch.tensor([0])
    for in_goal in (True, True, False, True, True):
        hold_steps = advance_goal_hold_steps(hold_steps, torch.tensor([in_goal]))
        assert not bool(has_held_goal(hold_steps, required).item())

    hold_steps = advance_goal_hold_steps(hold_steps, torch.tensor([True]))

    assert bool(has_held_goal(hold_steps, required).item())
