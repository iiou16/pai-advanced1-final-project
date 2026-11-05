"""通路のシーンの Go2（設定の台数）と箱を読み書きし、物理を進める。"""

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from cooperative_go2.entities.corridor import BoxFootprint
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import Go2State
from cooperative_go2.entities.planar_frame import yaw_from_quat_wxyz
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.interface_adapters.config_loader import CorridorSceneSettings
from cooperative_go2.interface_adapters.genesis_corridor_scene import (
    CorridorScene,
    CorridorSceneRequest,
    build_corridor_scene,
)
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body

PLANAR_AXES = slice(0, 2)
VERTICAL_AXIS_INDEX = 2
# 箱を置き直すときの姿勢（回転なし）の四元数 (w, x, y, z)
UPRIGHT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


class RobotCountMismatchError(ValueError):
    """ロボットごとに渡す値の数が、シーンに置いたロボットの数と違うことを表す。"""


def _read_planar(entity: RigidEntity) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=entity.get_pos()[:, PLANAR_AXES],
        yaw_rad=yaw_from_quat_wxyz(entity.get_quat()),
        lin_vel_xy_mps=entity.get_vel()[:, PLANAR_AXES],
        yaw_rate_radps=entity.get_ang()[:, VERTICAL_AXIS_INDEX],
    )


class GenesisCorridorSimulation:
    """通路のシーンの Go2 と箱を操作する。ロボットごとの値は、設定の robots と同じ順のタプルで受け渡す。

    前提: シーンは build 済み。
    """

    def __init__(self, scene: CorridorScene, settings: CorridorSceneSettings) -> None:
        """構築済みのシーンを受け取り、各 Go2 の関節を設定する。

        Raises:
            UnexpectedRobotModelError: Go2 のモデルが想定と異なる。
        """
        self._scene = scene
        self._layout = settings.scene
        self._bodies = tuple(
            GenesisGo2Body(robot, config) for robot, config in zip(scene.robots, scene.robot_configs, strict=True)
        )
        size_x, size_y, size_z = self._layout.box.size_m
        self._box_half_size_xy_m = (size_x / 2, size_y / 2)
        self._box_initial_position = (*self._layout.box.initial_xy_m, size_z / 2)

    @classmethod
    def build(
        cls, gait_config: GaitConfig, settings: CorridorSceneSettings, request: CorridorSceneRequest
    ) -> "GenesisCorridorSimulation":
        """通路のシーンを構築し、Go2 の関節を設定する。前提: gs.init() 済みであること。

        Raises:
            ValueError: num_envs が 1 未満。
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
        """
        return cls(build_corridor_scene(gait_config, settings, request), settings)

    @property
    def scene(self) -> CorridorScene:
        """構築済みのシーン（評価動画のカメラを取り出すのに使う）。"""
        return self._scene

    @property
    def robot_names(self) -> tuple[str, ...]:
        """ロボットの名前。設定の robots と同じ順。"""
        return tuple(placement.name for placement in self._layout.robots)

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (12,) [rad]。全ロボットで共通。"""
        return self._bodies[0].default_dof_pos

    def read_go2_states(self) -> tuple[Go2State, ...]:
        """各 Go2 の現在の状態を返す。"""
        return tuple(body.read_state() for body in self._bodies)

    def read_robot_planars(self) -> tuple[PlanarBodyState, ...]:
        """各 Go2 の胴体の水平面の状態（ワールド座標系）を返す。"""
        return tuple(_read_planar(robot) for robot in self._scene.robots)

    def read_box_planar(self) -> PlanarBodyState:
        """箱の水平面の状態（ワールド座標系）を返す。"""
        return _read_planar(self._scene.box)

    def read_box_footprint(self) -> BoxFootprint:
        """箱を真上から見た長方形を返す。通路の開通の判定に使う。"""
        box = self.read_box_planar()
        return BoxFootprint(
            center_xy_m=box.position_xy_m, yaw_rad=box.yaw_rad, half_size_xy_m=self._box_half_size_xy_m
        )

    def robot_box_contact_masks(self) -> tuple[torch.Tensor, ...]:
        """各 Go2 が箱に触れている環境のマスク (num_envs,) を返す。"""
        return tuple(self._scene.robot_box_contact_mask(index) for index in range(len(self._bodies)))

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        """各 Go2 の PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。

        Raises:
            RobotCountMismatchError: joint_targets の数がロボットの数と違う。
        """
        self._require_one_per_robot(len(joint_targets))
        for body, targets in zip(self._bodies, joint_targets, strict=True):
            body.apply_joint_targets(targets)

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        self._scene.step()

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで物理計算が発散した環境のマスク (num_envs,) を返す。"""
        return self._scene.diverged_mask()

    def set_box_mass(self, mass_kg: float) -> None:
        """全環境の箱の質量を mass_kg [kg] に変える（慣性も同じ比で変わる）。質量を変えて比べる計測に使う。"""
        self._scene.box.set_mass(mass_kg)

    def reset(self, mask: torch.Tensor, dof_pos: tuple[torch.Tensor, ...]) -> None:
        """mask が True の環境の各 Go2 を初期位置・関節角 dof_pos で、箱を初期位置に水平に置く。

        箱の位置を書き換えるのはエピソードの始めのこのときだけで、エピソードの間は接触と摩擦だけで動かす。

        Args:
            mask: リセットする環境のマスク (num_envs,)。
            dof_pos: 各 Go2 のリセット後の関節角 (num_envs, 12) [rad]。mask が False の行は使わない。

        Raises:
            RobotCountMismatchError: dof_pos の数がロボットの数と違う。
        """
        self._require_one_per_robot(len(dof_pos))
        for body, robot_dof_pos in zip(self._bodies, dof_pos, strict=True):
            body.reset(mask, robot_dof_pos)
        num_envs = mask.shape[0]
        box = self._scene.box
        position = torch.tensor(self._box_initial_position, dtype=gs.tc_float, device=gs.device)
        box.set_pos(position.expand(num_envs, -1), envs_idx=mask, zero_velocity=True)
        upright = torch.tensor(UPRIGHT_QUAT_WXYZ, dtype=gs.tc_float, device=gs.device)
        box.set_quat(upright.expand(num_envs, -1), envs_idx=mask, zero_velocity=True)

    def _require_one_per_robot(self, count: int) -> None:
        if count != len(self._bodies):
            raise RobotCountMismatchError(f"ロボット {len(self._bodies)} 台分の値が必要: {count} 個")
