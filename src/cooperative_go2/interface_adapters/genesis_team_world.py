"""通路のシーン（並列環境 1 つ）を、ROS 2 のシミュレーションノードの試行の流れが使うシーン（SimulatedTeamWorld）として見せる。"""

from dataclasses import fields

import torch

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.use_cases.ros_team_trial import TeamWorldState

CPU = torch.device("cpu")


class GenesisTeamWorld:
    """並列環境 1 つで構築した通路のシーンを操作する。読んだ状態は CPU に移して返す。"""

    def __init__(self, simulation: GenesisCorridorSimulation) -> None:
        """並列環境 1 つで構築したシミュレーションを受け取る。"""
        self._simulation = simulation
        self._all_envs = torch.ones(1, dtype=torch.bool, device=simulation.default_dof_pos.device)

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (1, 12) [rad]（CPU）。"""
        return self._simulation.default_dof_pos.unsqueeze(0).to(CPU)

    def reset(self, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]) -> None:
        """各ロボットを開始位置 robot_xy_m (1, 2)・関節角 dof_pos (1, 12) で、箱を初期位置に置く。"""
        device = self._all_envs.device
        self._simulation.reset_at(
            self._all_envs, tuple(pos.to(device) for pos in dof_pos), tuple(xy.to(device) for xy in robot_xy_m)
        )

    def read(self) -> TeamWorldState:
        """現在の状態を CPU に移して返す。"""
        simulation = self._simulation
        return TeamWorldState(
            go2=tuple(_go2_state_on_cpu(state) for state in simulation.read_go2_states()),
            robots=tuple(_pose_on_cpu(pose) for pose in simulation.read_robot_poses()),
            box=_pose_on_cpu(simulation.read_box_pose()),
        )

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        """各ロボットの PD 制御の目標関節角 (1, 12) [rad] を設定する。"""
        device = self._all_envs.device
        self._simulation.apply_joint_targets(tuple(targets.to(device) for targets in joint_targets))

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        self._simulation.step()

    def diverged(self) -> bool:
        """直前の物理ステップで物理計算が発散したか。"""
        return bool(self._simulation.diverged_mask()[0])


def _pose_on_cpu(pose: RigidBodyPose) -> RigidBodyPose:
    return RigidBodyPose(**{field.name: getattr(pose, field.name).to(CPU) for field in fields(pose)})


def _go2_state_on_cpu(state: Go2State) -> Go2State:
    proprioception = Go2Proprioception(
        **{field.name: getattr(state.proprioception, field.name).to(CPU) for field in fields(state.proprioception)}
    )
    return Go2State(
        proprioception=proprioception,
        base_pos_m=state.base_pos_m.to(CPU),
        base_euler_deg=state.base_euler_deg.to(CPU),
        base_lin_vel=state.base_lin_vel.to(CPU),
    )
