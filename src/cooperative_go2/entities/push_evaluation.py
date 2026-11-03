"""箱押しの評価指標。エピソードの開始時と終了時の箱の状態から、目標への進み具合と箱の回転・傾きを求める。"""

import torch

from cooperative_go2.entities.planar_frame import wrap_angle_rad


class UndefinedGoalDirectionError(ValueError):
    """箱の開始位置が目標位置と一致し、目標方向が定まらないことを表す。"""


def goal_distance_m(box_xy_m: torch.Tensor, goal_xy_m: torch.Tensor) -> torch.Tensor:
    """箱の中心から目標位置までの水平距離 (num_envs,) [m] を返す。

    Args:
        box_xy_m: 箱の中心 (num_envs, 2)、ワールド座標系 [m]。
        goal_xy_m: 目標位置 (num_envs, 2)、ワールド座標系 [m]。
    """
    return torch.linalg.vector_norm(goal_xy_m - box_xy_m, dim=1)


def progress_toward_goal_m(
    box_start_xy_m: torch.Tensor, box_end_xy_m: torch.Tensor, goal_xy_m: torch.Tensor
) -> torch.Tensor:
    """箱が、開始位置から見た目標の方向へ進んだ量 (num_envs,) [m] を返す。

    箱の変位を「開始位置 → 目標」の単位ベクトルに射影した値で、目標から遠ざかれば負になる。
    目標を通り過ぎた分も進んだ量に含む（最終的な目標との距離は goal_distance_m で別に測る）。

    Args:
        box_start_xy_m: エピソード開始時の箱の中心 (num_envs, 2)、ワールド座標系 [m]。
        box_end_xy_m: エピソード終了時の箱の中心 (num_envs, 2)、ワールド座標系 [m]。
        goal_xy_m: 目標位置 (num_envs, 2)、ワールド座標系 [m]。

    Raises:
        UndefinedGoalDirectionError: いずれかの環境で、箱の開始位置と目標位置が一致している。
    """
    start_to_goal = goal_xy_m - box_start_xy_m
    start_goal_distance = torch.linalg.vector_norm(start_to_goal, dim=1)
    coincident = start_goal_distance == 0
    if bool(coincident.any()):
        raise UndefinedGoalDirectionError(
            f"箱の開始位置が目標位置と一致している環境: {torch.nonzero(coincident).flatten().tolist()}"
        )
    displacement = box_end_xy_m - box_start_xy_m
    return torch.sum(displacement * start_to_goal, dim=1) / start_goal_distance


def yaw_change_deg(start_yaw_rad: torch.Tensor, end_yaw_rad: torch.Tensor) -> torch.Tensor:
    """開始時から終了時までの箱のヨー角の変化の大きさ (num_envs,) を 0 以上 180 以下の度で返す。

    一周以上回った場合は区別できず、残りの角度だけを返す（開始と終了の向きの差を測る指標のため）。

    Args:
        start_yaw_rad: エピソード開始時の箱のヨー角 (num_envs,) [rad]。
        end_yaw_rad: エピソード終了時の箱のヨー角 (num_envs,) [rad]。
    """
    return torch.rad2deg(torch.abs(wrap_angle_rad(end_yaw_rad - start_yaw_rad)))


def tilt_deg(up_axis_world_z: torch.Tensor) -> torch.Tensor:
    """箱の上向きの軸がワールドの鉛直上向きから傾いた角度 (num_envs,) を 0 以上 180 以下の度で返す。

    姿勢をオイラー角で受け取らないのは、オイラー角の回転順の取り決めに結果が左右されないようにするため。

    Args:
        up_axis_world_z: 箱の上向きの単位ベクトルをワールド座標系で表したときの z 成分 (num_envs,)。
            1 で水平に置かれた状態、0 で横倒し。丸め誤差で [-1, 1] をわずかに超えた値は端に寄せる。
    """
    return torch.rad2deg(torch.acos(torch.clamp(up_axis_world_z, -1.0, 1.0)))
