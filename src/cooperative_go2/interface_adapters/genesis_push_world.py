"""シーン S1_push（並列環境 1 つ）を、ROS 2 のシミュレーションノードの試行の流れが使うシーン（SimulatedPushWorld）として見せる。"""

from dataclasses import fields

import torch

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State
from cooperative_go2.entities.push_layout import PushLayout
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.interface_adapters.genesis_push_simulation import GenesisPushSimulation
from cooperative_go2.use_cases.ros_push_trial import PushWorldState

CPU = torch.device("cpu")


class GenesisPushWorld:
    """並列環境 1 つで構築したシーン S1_push を操作する。読んだ状態は CPU に移して返す。"""

    def __init__(self, simulation: GenesisPushSimulation) -> None:
        """並列環境 1 つで構築したシミュレーションを受け取る。"""
        self._simulation = simulation
        self._all_envs = torch.ones(1, dtype=torch.bool, device=simulation.default_dof_pos.device)

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (1, 12) [rad]（CPU）。"""
        return self._simulation.default_dof_pos.unsqueeze(0).to(CPU)

    def reset(self, layout: PushLayout, dof_pos: torch.Tensor) -> None:
        """Go2 を初期位置・関節角 dof_pos (1, 12) で、箱と目標を layout (1 行) の位置に置く。"""
        device = self._all_envs.device
        layout_on_device = PushLayout(box_xy_m=layout.box_xy_m.to(device), goal_xy_m=layout.goal_xy_m.to(device))
        self._simulation.reset(self._all_envs, dof_pos.to(device), layout_on_device)

    def read(self) -> PushWorldState:
        """現在の状態を CPU に移して返す。"""
        simulation = self._simulation
        return PushWorldState(
            go2=_go2_state_on_cpu(simulation.read_go2_state()),
            robot=_pose_on_cpu(simulation.read_robot_pose()),
            box=_pose_on_cpu(simulation.read_box_pose()),
            box_contact=simulation.robot_box_contact_mask().to(CPU),
        )

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (1, 12) [rad] を設定する。"""
        self._simulation.apply_joint_targets(joint_targets.to(self._all_envs.device))

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
