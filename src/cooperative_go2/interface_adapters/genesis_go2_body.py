"""Genesis の Go2 エンティティと、Go2State・目標関節角との受け渡し。

シーンの構築とは分けてあり、Go2 のエンティティだけに依存する。
"""

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidJoint
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from cooperative_go2.entities.gait_config import Go2RobotConfig
from cooperative_go2.entities.go2 import GO2_JOINT_NAMES, NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State, initial_go2_state
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

# 浮遊する胴体の一般化座標は、位置 3 ＋ 姿勢の四元数 4 で qpos の先頭に並ぶ
BASE_QPOS_SIZE = 7
GO2_QPOS_SIZE = BASE_QPOS_SIZE + NUM_GO2_JOINTS
BASE_POS_SLICE = slice(0, 3)
BASE_QUAT_SLICE = slice(3, 7)
GRAVITY_DIRECTION = (0.0, 0.0, -1.0)


class UnexpectedRobotModelError(SimulationSetupError):
    """読み込んだ Go2 のモデルが、前提とする構成（浮遊胴体 ＋ 1 自由度の関節 12 個）と異なることを表す。"""


class GenesisGo2Body:
    """シーンに置いた Go2 の状態を読み、目標関節角を与え、初期姿勢に戻す。

    前提: robot を含むシーンが build 済みであること。関節は GO2_JOINT_NAMES の順で扱う。
    """

    def __init__(self, robot: RigidEntity, robot_config: Go2RobotConfig) -> None:
        """関節の対応を確かめ、PD ゲインを設定する。

        Args:
            robot: build 済みのシーンに置いた Go2 のエンティティ。
            robot_config: 既定姿勢・PD ゲイン・初期位置。

        Raises:
            UnexpectedRobotModelError: Go2 のモデルに必要な関節が無い、または自由度が想定と異なる。
        """
        self._robot = robot
        try:
            joints = self._find_motor_joints()
            self._motor_dofs_idx = [joint.dofs_idx_local[0] for joint in joints]
            robot.set_dofs_kp([robot_config.kp] * NUM_GO2_JOINTS, self._motor_dofs_idx)
            robot.set_dofs_kv([robot_config.kd] * NUM_GO2_JOINTS, self._motor_dofs_idx)
        except gs.GenesisException as error:
            raise UnexpectedRobotModelError(f"Go2 の関節を設定できない: {error}") from error

        def tensor(values: object) -> torch.Tensor:
            return torch.tensor(values, dtype=gs.tc_float, device=gs.device)

        default_angles = robot_config.default_joint_angles_rad.in_order()
        self.default_dof_pos = tensor(default_angles)
        init_base_pos = tensor(robot_config.base_init_pos_m)
        init_base_quat = tensor(robot_config.base_init_quat_wxyz)
        self._inv_init_base_quat = inv_quat(init_base_quat)
        self._gravity = tensor(GRAVITY_DIRECTION)
        self.initial_state = initial_go2_state(
            init_base_pos, transform_by_quat(self._gravity, self._inv_init_base_quat), self.default_dof_pos
        )
        self._init_qpos = torch.zeros(GO2_QPOS_SIZE, dtype=gs.tc_float, device=gs.device)
        self._init_qpos[BASE_POS_SLICE] = init_base_pos
        self._init_qpos[BASE_QUAT_SLICE] = init_base_quat
        # 関節角を GO2_JOINT_NAMES の順に qpos へ書き込むための、各関節の qpos 内の位置
        self._joint_qs_idx = torch.tensor([joint.qs_idx_local[0] for joint in joints], device=gs.device)
        self._init_qpos[self._joint_qs_idx] = self.default_dof_pos

    def _find_motor_joints(self) -> list[RigidJoint]:
        joints = [self._robot.get_joint(name) for name in GO2_JOINT_NAMES]
        if self._robot.n_qs != GO2_QPOS_SIZE or any(joint.n_dofs != 1 for joint in joints):
            raise UnexpectedRobotModelError(
                f"Go2 の qpos の次元 {self._robot.n_qs}（想定 {GO2_QPOS_SIZE}）、"
                f"関節の自由度 {[joint.n_dofs for joint in joints]}"
            )
        return joints

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

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の Go2 を初期位置で、関節角を dof_pos にして静止させる。

        Args:
            mask: リセットする環境のマスク (num_envs,)。
            dof_pos: リセット後の関節角 (num_envs, 12) [rad]、GO2_JOINT_NAMES の順。mask が False の行は使わない。
        """
        qpos = self._init_qpos.expand(dof_pos.shape[0], -1).clone()
        qpos[:, self._joint_qs_idx] = dof_pos
        self._robot.set_qpos(qpos, envs_idx=mask, zero_velocity=True, skip_forward=True)
