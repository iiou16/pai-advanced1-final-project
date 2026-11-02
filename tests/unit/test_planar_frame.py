import math

import pytest
import torch

from cooperative_go2.entities.planar_frame import relative_xy_in_robot_frame, rotate_world_to_robot_xy, wrap_angle_rad


def test_wrap_angle_maps_into_half_open_range() -> None:
    angles = torch.tensor([0.0, math.pi, -math.pi, 3 * math.pi / 2, -3 * math.pi / 2, 4 * math.pi + 0.1])

    wrapped = wrap_angle_rad(angles)

    expected = torch.tensor([0.0, -math.pi, -math.pi, -math.pi / 2, math.pi / 2, 0.1])
    assert torch.allclose(wrapped, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("robot_yaw_rad", "expected_xy"),
    [
        # ロボットが +x を向いていれば、ワールドの +x はロボットの前
        (0.0, (1.0, 0.0)),
        # ロボットが +y（左）を向けば、ワールドの +x はロボットの右（-y）
        (math.pi / 2, (0.0, -1.0)),
        # ロボットが -x を向けば、ワールドの +x はロボットの後ろ
        (math.pi, (-1.0, 0.0)),
    ],
)
def test_rotate_world_to_robot_expresses_world_x_axis_in_robot_frame(
    robot_yaw_rad: float, expected_xy: tuple[float, float]
) -> None:
    world_x_axis = torch.tensor([[1.0, 0.0]])

    in_robot_frame = rotate_world_to_robot_xy(world_x_axis, torch.tensor([robot_yaw_rad]))

    assert torch.allclose(in_robot_frame, torch.tensor([expected_xy]), atol=1e-6)


def test_rotate_world_to_robot_handles_each_env_with_its_own_yaw() -> None:
    vectors = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    yaws = torch.tensor([0.0, math.pi / 2])

    in_robot_frame = rotate_world_to_robot_xy(vectors, yaws)

    assert torch.allclose(in_robot_frame, torch.tensor([[0.0, 2.0], [2.0, 0.0]]), atol=1e-6)


def test_relative_position_subtracts_robot_position_before_rotating() -> None:
    target = torch.tensor([[3.0, 4.0]])
    robot = torch.tensor([[3.0, 2.0]])

    # 対象はロボットからワールドの +y に 2 m。ロボットは +y を向いているので真正面に見える
    relative = relative_xy_in_robot_frame(target, robot, torch.tensor([math.pi / 2]))

    assert torch.allclose(relative, torch.tensor([[2.0, 0.0]]), atol=1e-6)
