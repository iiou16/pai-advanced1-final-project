"""シーン S1_push の Genesis 実装を、箱押しの環境が使うシミュレーションの操作（PushSimulation）として提供する。"""

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import Go2State
from cooperative_go2.entities.planar_frame import yaw_from_quat_wxyz
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.interface_adapters.config_loader import PushSceneSettings
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body
from cooperative_go2.interface_adapters.genesis_push_scene import PushScene, build_push_scene

PLANAR_AXES = slice(0, 2)
VERTICAL_AXIS_INDEX = 2
# 箱を置き直すときの姿勢（回転なし）の四元数 (w, x, y, z)
UPRIGHT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _read_planar(entity: RigidEntity) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=entity.get_pos()[:, PLANAR_AXES],
        yaw_rad=yaw_from_quat_wxyz(entity.get_quat()),
        lin_vel_xy_mps=entity.get_vel()[:, PLANAR_AXES],
        yaw_rate_radps=entity.get_ang()[:, VERTICAL_AXIS_INDEX],
    )


class GenesisPushSimulation:
    """シーン S1_push の Go2 と箱を読み書きし、物理を進める。前提: シーンは build 済み。"""

    def __init__(self, scene: PushScene, body: GenesisGo2Body, box_half_height_m: float) -> None:
        """構築済みのシーンと Go2 の操作を受け取る。

        Args:
            scene: 構築済みのシーン S1_push。
            body: scene の Go2 を操作するもの。
            box_half_height_m: 箱の高さの半分 [m]。箱を床に置き直すときの中心の高さに使う。
        """
        self._scene = scene
        self._body = body
        self._box_half_height_m = box_half_height_m

    @classmethod
    def build(cls, gait_config: GaitConfig, settings: PushSceneSettings, num_envs: int) -> "GenesisPushSimulation":
        """シーン S1_push を構築し、Go2 の関節を設定する。前提: gs.init() 済みであること。

        Raises:
            ValueError: num_envs が 1 未満。
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
        """
        scene = build_push_scene(gait_config, settings, num_envs)
        body = GenesisGo2Body(scene.robot, gait_config.robot)
        return cls(scene, body, box_half_height_m=settings.box.size_m[VERTICAL_AXIS_INDEX] / 2)

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (12,) [rad]。"""
        return self._body.default_dof_pos

    @property
    def initial_go2_state(self) -> Go2State:
        """リセット直後の Go2 の状態（全環境に共通の 1 行）。"""
        return self._body.initial_state

    def read_go2_state(self) -> Go2State:
        """Go2 の現在の状態を返す。"""
        return self._body.read_state()

    def read_robot_planar(self) -> PlanarBodyState:
        """Go2 の胴体の水平面の状態（ワールド座標系）を返す。"""
        return _read_planar(self._scene.robot)

    def read_box_planar(self) -> PlanarBodyState:
        """箱の水平面の状態（ワールド座標系）を返す。"""
        return _read_planar(self._scene.box)

    def robot_box_contact_mask(self) -> torch.Tensor:
        """Go2 が箱に触れている環境のマスク (num_envs,) を返す。"""
        return self._scene.robot_box_contact_mask()

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。"""
        self._body.apply_joint_targets(joint_targets)

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        self._scene.step()

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで物理計算が発散した環境のマスク (num_envs,) を返す。"""
        return self._scene.diverged_mask()

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor, box_xy_m: torch.Tensor) -> None:
        """mask が True の環境の Go2 を初期位置・関節角 dof_pos で、箱を box_xy_m (num_envs, 2) に水平に置く。

        箱の位置を書き換えるのはエピソードの始めのこのときだけで、エピソードの間は接触と摩擦だけで動かす（要件 3.1）。
        """
        self._body.reset(mask, dof_pos)
        num_envs = box_xy_m.shape[0]
        heights = torch.full((num_envs, 1), self._box_half_height_m, dtype=box_xy_m.dtype, device=box_xy_m.device)
        box = self._scene.box
        box.set_pos(torch.cat((box_xy_m, heights), dim=1), envs_idx=mask, zero_velocity=True)
        upright = torch.tensor(UPRIGHT_QUAT_WXYZ, dtype=gs.tc_float, device=gs.device).expand(num_envs, -1)
        box.set_quat(upright, envs_idx=mask, zero_velocity=True)
