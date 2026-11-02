"""水平面での座標変換。ワールド座標系の xy 成分を、ロボットのヨー角だけ回したロボット座標系へ移す。

箱押しは平らな床の上で行うため、高レベル方策の観測と評価は水平面の成分だけで扱う。
角度はすべて上から見て反時計回りが正 [rad]。テンソルは並列環境のバッチ (num_envs, ...)。
"""

import math

import torch


def wrap_angle_rad(angle_rad: torch.Tensor) -> torch.Tensor:
    """角度を [-π, π) に折り返した値を返す。"""
    return torch.remainder(angle_rad + math.pi, 2 * math.pi) - math.pi


def rotate_world_to_robot_xy(vectors_world_xy: torch.Tensor, robot_yaw_rad: torch.Tensor) -> torch.Tensor:
    """ワールド座標系の水平ベクトルを、ロボット座標系（前が +x、左が +y）で表したものを返す。

    Args:
        vectors_world_xy: ワールド座標系の水平ベクトル (num_envs, 2)。速度など、位置の差を取った後の量を渡す。
        robot_yaw_rad: ロボットのヨー角 (num_envs,)。
    """
    cos_yaw = torch.cos(robot_yaw_rad)
    sin_yaw = torch.sin(robot_yaw_rad)
    world_x = vectors_world_xy[:, 0]
    world_y = vectors_world_xy[:, 1]
    return torch.stack((cos_yaw * world_x + sin_yaw * world_y, -sin_yaw * world_x + cos_yaw * world_y), dim=1)


def relative_xy_in_robot_frame(
    target_world_xy: torch.Tensor, robot_world_xy: torch.Tensor, robot_yaw_rad: torch.Tensor
) -> torch.Tensor:
    """ロボットから見た対象の相対位置 (num_envs, 2) を、ロボット座標系で返す [m]。

    Args:
        target_world_xy: 対象の位置 (num_envs, 2)、ワールド座標系 [m]。
        robot_world_xy: ロボットの胴体の位置 (num_envs, 2)、ワールド座標系 [m]。
        robot_yaw_rad: ロボットのヨー角 (num_envs,)。
    """
    return rotate_world_to_robot_xy(target_world_xy - robot_world_xy, robot_yaw_rad)


def yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """姿勢の四元数 (num_envs, 4)（w, x, y, z の順）から、ヨー角 (num_envs,) [rad] を返す。

    胴体の前方の軸を水平面に射影した向きの角度。傾いていても、前方の軸が鉛直でなければ定まる。
    """
    w, x, y, z = quat_wxyz.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def up_axis_world_z_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """姿勢の四元数 (num_envs, 4)（w, x, y, z の順）で回した上向きの軸の、ワールドの z 成分 (num_envs,) を返す。

    1 で水平、0 で横倒し。傾きの角度は push_evaluation.tilt_deg で求める。
    """
    _, x, y, _ = quat_wxyz.unbind(dim=1)
    return 1 - 2 * (x * x + y * y)
