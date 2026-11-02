"""シーン S0_flat と Go2 を、歩行環境が使うシミュレーションの操作（GaitSimulation）にまとめる。"""

import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import Go2State
from cooperative_go2.interface_adapters.genesis_flat_scene import CameraSpec, FlatScene, build_flat_scene
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body


class GenesisGaitSimulation:
    """Genesis で構築したシーン S0_flat と、そこに置いた Go2 の操作。前提: gs.init() 済みであること。"""

    def __init__(self, scene: FlatScene, body: GenesisGo2Body) -> None:
        """構築済みのシーンと、そのシーンの Go2 を受け取る。"""
        self.scene = scene
        self._body = body

    @classmethod
    def build(cls, config: GaitConfig, camera_spec: CameraSpec | None) -> "GenesisGaitSimulation":
        """設定に従ってシーンを構築し、Go2 の関節と PD ゲインを設定する。

        Raises:
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
        """
        scene = build_flat_scene(config, camera_spec)
        return cls(scene, GenesisGo2Body(scene.robot, config.robot))

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """既定姿勢の関節角 (12,) [rad]。"""
        return self._body.default_dof_pos

    @property
    def initial_state(self) -> Go2State:
        """リセット直後の状態（全環境に共通の 1 行）。"""
        return self._body.initial_state

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。"""
        self._body.apply_joint_targets(joint_targets)

    def step(self) -> None:
        """物理を 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: 衝突候補・接触点の数が上限を超えた。
        """
        self.scene.step()

    def read_state(self) -> Go2State:
        """現在の Go2 の状態を返す。"""
        return self._body.read_state()

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで物理計算が発散した環境のマスク (num_envs,) を返す。"""
        return self.scene.diverged_mask()

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の Go2 を初期位置で、関節角を dof_pos (num_envs, 12) [rad] にして静止させる。"""
        self._body.reset(mask, dof_pos)
