import math

import pytest
import torch

from cooperative_go2.entities.planar_frame import (
    relative_xy_in_robot_frame,
    rotate_world_to_robot_xy,
    up_axis_world_z_from_quat_wxyz,
    wrap_angle_rad,
    yaw_from_quat_wxyz,
)


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


def _quat_about_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_about_x(roll: float) -> list[float]:
    return [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]


def test_yaw_from_quaternion_recovers_rotation_about_vertical_axis() -> None:
    yaws = [0.0, math.pi / 2, -2.5, 3.0]

    recovered = yaw_from_quat_wxyz(torch.tensor([_quat_about_z(yaw) for yaw in yaws]))

    assert torch.allclose(recovered, torch.tensor(yaws), atol=1e-6)


def test_yaw_is_unaffected_by_roll() -> None:
    roll = math.radians(20.0)
    yaw = 0.7
    # 先にロール、次にヨーで回した姿勢（ワールド座標で q = q_yaw * q_roll）
    w1, x1, y1, z1 = _quat_about_z(yaw)
    w2, x2, y2, z2 = _quat_about_x(roll)
    quat = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]

    assert yaw_from_quat_wxyz(torch.tensor([quat])).item() == pytest.approx(yaw, abs=1e-6)


def test_up_axis_z_is_cosine_of_tilt() -> None:
    quats = torch.tensor([_quat_about_z(1.0), _quat_about_x(math.radians(30.0)), _quat_about_x(math.pi / 2)])

    up_z = up_axis_world_z_from_quat_wxyz(quats)

    assert torch.allclose(up_z, torch.tensor([1.0, math.cos(math.radians(30.0)), 0.0]), atol=1e-6)
