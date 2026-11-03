"""剛体（Go2 の胴体・箱）の位置・姿勢・速度。ROS 2 の Odometry と同じく、速度を本体座標系で持つ。"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.planar_frame import yaw_from_quat_wxyz
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.quaternion import rotate_by_inverse_quat_wxyz, rotate_by_quat_wxyz

PLANAR_AXES = slice(0, 2)
VERTICAL_AXIS_INDEX = 2
GRAVITY_DIRECTION = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class RigidBodyPose:
    """剛体の位置・姿勢・速度。テンソルは並列環境のバッチ。

    Attributes:
        position_m: 位置 (num_envs, 3)、ワールド座標系 [m]。
        quat_wxyz: 姿勢の単位四元数 (num_envs, 4)、w, x, y, z の順。
        lin_vel_body_mps: 線速度 (num_envs, 3)、本体座標系 [m/s]。
        ang_vel_body_radps: 角速度 (num_envs, 3)、本体座標系 [rad/s]。
    """

    position_m: torch.Tensor
    quat_wxyz: torch.Tensor
    lin_vel_body_mps: torch.Tensor
    ang_vel_body_radps: torch.Tensor

    @classmethod
    def from_world_velocities(
        cls, position_m: torch.Tensor, quat_wxyz: torch.Tensor, lin_vel_world_mps: torch.Tensor,
        ang_vel_world_radps: torch.Tensor,
    ) -> "RigidBodyPose":  # fmt: skip
        """ワールド座標系の速度 (num_envs, 3) から、本体座標系の速度を持つ状態を作る。"""
        return cls(
            position_m=position_m,
            quat_wxyz=quat_wxyz,
            lin_vel_body_mps=rotate_by_inverse_quat_wxyz(lin_vel_world_mps, quat_wxyz),
            ang_vel_body_radps=rotate_by_inverse_quat_wxyz(ang_vel_world_radps, quat_wxyz),
        )

    def to_planar(self) -> PlanarBodyState:
        """水平面の状態（位置・ヨー角・ワールド座標系の水平速度・ヨー角速度）を返す。

        速度は本体座標系からワールド座標系に戻してから水平成分を取る。学習環境はワールド座標系の速度を
        シミュレータから直接読むので、傾いていても同じ値になるよう 3 次元のまま回す。
        """
        lin_vel_world = rotate_by_quat_wxyz(self.lin_vel_body_mps, self.quat_wxyz)
        ang_vel_world = rotate_by_quat_wxyz(self.ang_vel_body_radps, self.quat_wxyz)
        return PlanarBodyState(
            position_xy_m=self.position_m[:, PLANAR_AXES],
            yaw_rad=yaw_from_quat_wxyz(self.quat_wxyz),
            lin_vel_xy_mps=lin_vel_world[:, PLANAR_AXES],
            yaw_rate_radps=ang_vel_world[:, VERTICAL_AXIS_INDEX],
        )

    def projected_gravity(self) -> torch.Tensor:
        """本体座標系で表した重力方向の単位ベクトル (num_envs, 3) を返す。水平に立っていれば (0, 0, -1)。"""
        gravity = torch.tensor(GRAVITY_DIRECTION, dtype=self.quat_wxyz.dtype, device=self.quat_wxyz.device)
        return rotate_by_inverse_quat_wxyz(gravity.expand(self.quat_wxyz.shape[0], -1), self.quat_wxyz)


def proprioception_from_pose(base: RigidBodyPose, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Go2Proprioception:
    """胴体の状態と関節の状態から、歩行方策の観測に使う自己受容の状態を作る。

    Args:
        base: Go2 の胴体の状態。
        dof_pos: 関節角 (num_envs, 12) [rad]、GO2_JOINT_NAMES の順。
        dof_vel: 関節角速度 (num_envs, 12) [rad/s]、GO2_JOINT_NAMES の順。
    """
    return Go2Proprioception(
        base_ang_vel=base.ang_vel_body_radps,
        projected_gravity=base.projected_gravity(),
        dof_pos=dof_pos,
        dof_vel=dof_vel,
    )
