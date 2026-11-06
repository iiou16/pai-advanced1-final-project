"""姿勢の四元数（w, x, y, z の順）によるベクトルの回転。テンソルは並列環境のバッチ (num_envs, ...)。

シミュレータのライブラリに頼らずに本体座標系とワールド座標系の間を変換するために使う。
ROS 2 の推論ノードは Genesis を読み込まないため、座標変換をここで自前で持つ。
"""

import torch


def rotate_by_quat_wxyz(vectors: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """本体座標系のベクトル (num_envs, 3) を、姿勢 quat_wxyz (num_envs, 4) でワールド座標系に回して返す。

    quat_wxyz は単位四元数であること（正規化はしない）。
    """
    w = quat_wxyz[:, :1]
    axis = quat_wxyz[:, 1:]
    # v' = v + 2w (u × v) + 2 u × (u × v)。u は四元数のベクトル部
    cross = torch.linalg.cross(axis, vectors, dim=1)
    return vectors + 2 * w * cross + 2 * torch.linalg.cross(axis, cross, dim=1)


def rotate_by_inverse_quat_wxyz(vectors: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """ワールド座標系のベクトル (num_envs, 3) を、姿勢 quat_wxyz (num_envs, 4) の本体座標系で表して返す。

    quat_wxyz は単位四元数であること（正規化はしない）。
    """
    conjugate = torch.cat((quat_wxyz[:, :1], -quat_wxyz[:, 1:]), dim=1)
    return rotate_by_quat_wxyz(vectors, conjugate)


def roll_pitch_deg_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """姿勢 quat_wxyz (num_envs, 4) のロールとピッチ (num_envs, 2) [deg] を返す（Z-Y-X の順のオイラー角）。

    Go2 の初期姿勢は回転なしなので、初期姿勢から見た傾きと同じになる。転倒の判定に使う。
    quat_wxyz は単位四元数であること（正規化はしない）。
    """
    w, x, y, z = quat_wxyz.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return torch.rad2deg(torch.stack((roll, pitch), dim=1))
