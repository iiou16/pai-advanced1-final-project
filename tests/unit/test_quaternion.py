"""四元数によるベクトルの回転と、剛体の状態の変換（本体座標系 ↔ ワールド座標系）の検証。"""

import math

import pytest
import torch

from cooperative_go2.entities.quaternion import (
    roll_pitch_deg_from_quat_wxyz,
    rotate_by_inverse_quat_wxyz,
    rotate_by_quat_wxyz,
)
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose, proprioception_from_pose


def _quat_about_axis(axis: tuple[float, float, float], angle_rad: float) -> torch.Tensor:
    half = angle_rad / 2
    return torch.tensor([[math.cos(half), *(math.sin(half) * component for component in axis)]])


def _rotation_matrix_zyx(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    yaw_matrix = torch.tensor([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    pitch_matrix = torch.tensor([[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0], [-sin_p, 0.0, cos_p]])
    roll_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]])
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _quat_zyx(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    # ヨー → ピッチ → ロールの順に本体を回した姿勢（回転行列 Rz·Ry·Rx に対応する四元数）
    yaw_quat = _quat_about_axis((0.0, 0.0, 1.0), yaw)[0]
    pitch_quat = _quat_about_axis((0.0, 1.0, 0.0), pitch)[0]
    roll_quat = _quat_about_axis((1.0, 0.0, 0.0), roll)[0]
    return _quat_multiply(_quat_multiply(yaw_quat, pitch_quat), roll_quat).unsqueeze(0)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = left.tolist()
    w2, x2, y2, z2 = right.tolist()
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


YAW, PITCH, ROLL = 0.7, -0.2, 0.15


def test_rotation_matches_rotation_matrix() -> None:
    vector = torch.tensor([[0.3, -1.2, 0.5]])

    rotated = rotate_by_quat_wxyz(vector, _quat_zyx(YAW, PITCH, ROLL))

    expected = (_rotation_matrix_zyx(YAW, PITCH, ROLL) @ vector[0]).unsqueeze(0)
    assert rotated == pytest.approx(expected, abs=1e-6)


def test_inverse_rotation_undoes_rotation() -> None:
    vector = torch.tensor([[0.3, -1.2, 0.5]])
    quat = _quat_zyx(YAW, PITCH, ROLL)

    restored = rotate_by_inverse_quat_wxyz(rotate_by_quat_wxyz(vector, quat), quat)

    assert restored == pytest.approx(vector, abs=1e-6)


def test_planar_state_recovers_world_velocities_and_yaw() -> None:
    quat = _quat_zyx(YAW, PITCH, ROLL)
    lin_vel_world = torch.tensor([[0.4, -0.1, 0.02]])
    ang_vel_world = torch.tensor([[0.05, -0.03, 0.8]])
    pose = RigidBodyPose.from_world_velocities(torch.tensor([[1.0, 2.0, 0.3]]), quat, lin_vel_world, ang_vel_world)

    planar = pose.to_planar()

    assert planar.position_xy_m == pytest.approx(torch.tensor([[1.0, 2.0]]))
    assert planar.yaw_rad == pytest.approx(torch.tensor([YAW]), abs=1e-6)
    assert planar.lin_vel_xy_mps == pytest.approx(lin_vel_world[:, :2], abs=1e-6)
    assert planar.yaw_rate_radps == pytest.approx(ang_vel_world[:, 2], abs=1e-6)


def test_projected_gravity_is_world_down_in_body_frame() -> None:
    quat = _quat_zyx(YAW, PITCH, ROLL)
    pose = RigidBodyPose(torch.zeros(1, 3), quat, torch.zeros(1, 3), torch.zeros(1, 3))

    gravity = pose.projected_gravity()

    expected = (_rotation_matrix_zyx(YAW, PITCH, ROLL).T @ torch.tensor([0.0, 0.0, -1.0])).unsqueeze(0)
    assert gravity == pytest.approx(expected, abs=1e-6)


def test_proprioception_uses_body_angular_velocity_and_joint_state() -> None:
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ang_vel_body = torch.tensor([[0.1, 0.2, 0.3]])
    pose = RigidBodyPose(torch.zeros(1, 3), upright, torch.zeros(1, 3), ang_vel_body)
    dof_pos = torch.linspace(-1.0, 1.0, 12).unsqueeze(0)
    dof_vel = torch.linspace(2.0, -2.0, 12).unsqueeze(0)

    proprioception = proprioception_from_pose(pose, dof_pos, dof_vel)

    assert torch.equal(proprioception.base_ang_vel, ang_vel_body)
    assert proprioception.projected_gravity == pytest.approx(torch.tensor([[0.0, 0.0, -1.0]]))
    assert torch.equal(proprioception.dof_pos, dof_pos)
    assert torch.equal(proprioception.dof_vel, dof_vel)


def test_roll_pitch_from_quaternion() -> None:
    half = math.radians(20.0) / 2
    # x 軸まわりに 20°（ロール）、y 軸まわりに 20°（ピッチ）、z 軸まわりに 90°（ヨーは含めない）
    quats = torch.tensor(
        [
            [math.cos(half), math.sin(half), 0.0, 0.0],
            [math.cos(half), 0.0, math.sin(half), 0.0],
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
        ]
    )

    angles = roll_pitch_deg_from_quat_wxyz(quats)

    assert angles.tolist() == [pytest.approx([20.0, 0.0], abs=1e-4), pytest.approx([0.0, 20.0], abs=1e-4),
                               pytest.approx([0.0, 0.0], abs=1e-4)]  # fmt: skip
