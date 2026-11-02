"""Genesis の Go2 エンティティと、Go2State・目標関節角との受け渡し。

シーンの構築とは独立しており、箱などを含む別のシーンに置いた Go2 にも使える。
"""

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from cooperative_go2.entities.gait_config import Go2RobotConfig
from cooperative_go2.entities.go2 import GO2_JOINT_NAMES, NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State, initial_go2_state

# 浮遊する胴体の一般化座標は、位置 3 ＋ 姿勢の四元数 4 で qpos の先頭に並ぶ
BASE_QPOS_SIZE = 7
BASE_POS_SLICE = slice(0, 3)
BASE_QUAT_SLICE = slice(3, 7)
GRAVITY_DIRECTION = (0.0, 0.0, -1.0)


class UnexpectedRobotModelError(RuntimeError):
    """読み込んだ Go2 のモデルが、前提とする構成（浮遊胴体 ＋ 1 自由度の関節 12 個）と異なることを表す。"""


class GenesisGo2Body:
    """シーンに置いた Go2 の状態を読み、目標関節角を与え、初期姿勢に戻す。

    前提: robot を含むシーンが build 済みであること。関節は GO2_JOINT_NAMES の順で扱う。
    """

    def __init__(self, robot: RigidEntity, robot_config: Go2RobotConfig) -> None:
        """関節の対応を確かめ、PD ゲインを設定する。

        Raises:
            UnexpectedRobotModelError: Go2 のモデルの自由度が想定と異なる。
        """
        self._robot = robot
        joints = [robot.get_joint(name) for name in GO2_JOINT_NAMES]
        expected_qs = BASE_QPOS_SIZE + NUM_GO2_JOINTS
        if robot.n_qs != expected_qs or any(joint.n_dofs != 1 for joint in joints):
            raise UnexpectedRobotModelError(
                f"Go2 の qpos の次元 {robot.n_qs}（想定 {expected_qs}）、関節の自由度 {[joint.n_dofs for joint in joints]}"
            )
        self._motor_dofs_idx = [joint.dofs_idx_local[0] for joint in joints]
        robot.set_dofs_kp([robot_config.kp] * NUM_GO2_JOINTS, self._motor_dofs_idx)
        robot.set_dofs_kv([robot_config.kd] * NUM_GO2_JOINTS, self._motor_dofs_idx)

        def tensor(values: object) -> torch.Tensor:
            return torch.tensor(values, dtype=gs.tc_float, device=gs.device)

        self.default_dof_pos = tensor(robot_config.default_joint_angles_rad.in_order())
        init_base_pos = tensor(robot_config.base_init_pos_m)
        init_base_quat = tensor(robot_config.base_init_quat_wxyz)
        self._inv_init_base_quat = inv_quat(init_base_quat)
        self._gravity = tensor(GRAVITY_DIRECTION)
        self.initial_state = initial_go2_state(
            init_base_pos, transform_by_quat(self._gravity, self._inv_init_base_quat), self.default_dof_pos
        )
        self._init_qpos = torch.zeros(expected_qs, dtype=gs.tc_float, device=gs.device)
        self._init_qpos[BASE_POS_SLICE] = init_base_pos
        self._init_qpos[BASE_QUAT_SLICE] = init_base_quat
        for joint, angle in zip(joints, robot_config.default_joint_angles_rad.in_order(), strict=True):
            self._init_qpos[joint.qs_idx_local[0]] = angle

    def read_state(self) -> Go2State:
        """現在の状態を読み、速度と重力方向を本体座標系に直して返す。"""
        base_quat = self._robot.get_quat()
        inv_base_quat = inv_quat(base_quat)
        return Go2State(
            proprioception=Go2Proprioception(
                base_ang_vel=transform_by_quat(self._robot.get_ang(), inv_base_quat),
                projected_gravity=transform_by_quat(self._gravity, inv_base_quat),
                dof_pos=self._robot.get_dofs_position(self._motor_dofs_idx),
                dof_vel=self._robot.get_dofs_velocity(self._motor_dofs_idx),
            ),
            base_pos_m=self._robot.get_pos(),
            base_euler_deg=quat_to_xyz(
                transform_quat_by_quat(self._inv_init_base_quat, base_quat), rpy=True, degrees=True
            ),
            base_lin_vel=transform_by_quat(self._robot.get_vel(), inv_base_quat),
        )

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。次の物理ステップから効く。"""
        self._robot.control_dofs_position(joint_targets, self._motor_dofs_idx)

    def reset(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の Go2 を初期位置・既定姿勢で静止させる。"""
        self._robot.set_qpos(self._init_qpos, envs_idx=mask, zero_velocity=True, skip_forward=True)
