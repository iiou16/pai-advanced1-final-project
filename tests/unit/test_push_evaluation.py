import math

import pytest
import torch

from cooperative_go2.entities.push_evaluation import (
    UndefinedGoalDirectionError,
    goal_distance_m,
    progress_toward_goal_m,
    tilt_deg,
    yaw_change_deg,
)


def test_goal_distance_is_horizontal_euclidean_distance() -> None:
    distance = goal_distance_m(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([[3.0, 4.0], [1.0, 1.0]]))

    assert torch.allclose(distance, torch.tensor([5.0, 0.0]))


def test_progress_projects_displacement_onto_start_to_goal_direction() -> None:
    start = torch.zeros(4, 2)
    goal = torch.tensor([[2.0, 0.0]] * 4)
    end = torch.tensor([[1.0, 0.0], [1.0, 5.0], [-0.5, 0.0], [3.0, 0.0]])

    progress = progress_toward_goal_m(start, end, goal)

    # 目標方向への成分だけを数える。横へのずれは含まず、後退は負、通り過ぎた分も含む
    assert torch.allclose(progress, torch.tensor([1.0, 1.0, -0.5, 3.0]))


def test_progress_is_zero_when_box_does_not_move() -> None:
    start = torch.tensor([[1.0, 2.0]])

    progress = progress_toward_goal_m(start, start.clone(), torch.tensor([[4.0, 6.0]]))

    assert progress.tolist() == [0.0]


def test_progress_raises_when_start_coincides_with_goal() -> None:
    start = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    goal = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    with pytest.raises(UndefinedGoalDirectionError, match=r"\[1\]"):
        progress_toward_goal_m(start, start.clone(), goal)


def test_yaw_change_takes_shorter_way_around() -> None:
    start = torch.tensor([0.0, math.radians(170.0), math.radians(-90.0)])
    end = torch.tensor([math.radians(30.0), math.radians(-170.0), math.radians(90.0)])

    change = yaw_change_deg(start, end)

    # 170° → -170° は 340° ではなく 20° の回転
    assert torch.allclose(change, torch.tensor([30.0, 20.0, 180.0]), atol=1e-4)


def test_tilt_is_angle_between_box_up_axis_and_world_up() -> None:
    up_z = torch.tensor([1.0, math.cos(math.radians(30.0)), 0.0, 1.0 + 1e-7])

    tilt = tilt_deg(up_z)

    # 水平で 0°、横倒しで 90°。丸め誤差で 1 を超えても NaN にしない
    assert torch.allclose(tilt, torch.tensor([0.0, 30.0, 90.0, 0.0]), atol=1e-3)
