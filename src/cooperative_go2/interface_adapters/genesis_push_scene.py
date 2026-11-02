"""シーン S1_push（平面 ＋ Go2 ×1 ＋ 箱）を Genesis で構築し、物理ステップと、Go2 と箱の接触の取得を担う。

床と Go2 の置き方・物理の時間刻み・摩擦は歩行の学習時のシーン（S0_flat）と同じにする。
固定した歩行方策を、学習したときと同じ条件で動かすため。
"""

from dataclasses import dataclass

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.interface_adapters.config_loader import PushSceneSettings
from cooperative_go2.interface_adapters.genesis_flat_scene import ENABLE_SELF_COLLISION, GO2_URDF, PLANE_URDF, FlatScene
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

# 接触の照会で、相手のエンティティとの接触の有無を表すキー（Genesis の get_contacts の戻り値）
CONTACT_VALID_MASK_KEY = "valid_mask"


@dataclass(frozen=True)
class PushScene:
    """構築済みのシーン S1_push。

    床と Go2 の部分（物理ステップと、容量の超過・NaN の検出を含む）は FlatScene をそのまま使い、箱だけを足す。

    Attributes:
        ground_and_robot: 床と Go2。物理ステップと異常の検出もここで行う。
        ground: 床のエンティティ。
        box: 箱のエンティティ。動的な剛体で、Go2 との接触と床との摩擦だけで動く。
    """

    ground_and_robot: FlatScene
    ground: RigidEntity
    box: RigidEntity

    @property
    def robot(self) -> RigidEntity:
        """Go2 のエンティティ。"""
        return self.ground_and_robot.robot

    def step(self) -> None:
        """物理を 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: いずれかの環境で衝突候補・接触点の数が上限を超えた。
        """
        self.ground_and_robot.step()

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで NaN が出た（物理計算が発散した）環境を True とするマスク (num_envs,) を返す。"""
        return self.ground_and_robot.diverged_mask()

    def robot_box_contact_mask(self) -> torch.Tensor:
        """Go2 のいずれかのリンクが箱に触れている環境を True とするマスク (num_envs,) を返す。"""
        contacts = self.robot.get_contacts(with_entity=self.box)
        return contacts[CONTACT_VALID_MASK_KEY].any(dim=-1)

    def box_ground_contact_mask(self) -> torch.Tensor:
        """箱が床に触れている環境を True とするマスク (num_envs,) を返す。False なら箱が浮いている。"""
        contacts = self.box.get_contacts(with_entity=self.ground)
        return contacts[CONTACT_VALID_MASK_KEY].any(dim=-1)


def build_push_scene(gait_config: GaitConfig, settings: PushSceneSettings, num_envs: int) -> PushScene:
    """平面・Go2・箱を置いたシーンを num_envs 個の並列環境で構築する。

    前提: gs.init() 済みであること。

    Args:
        gait_config: 歩行方策を学習したときの設定。制御周期・サブステップ数と Go2 の初期位置・姿勢を使う。
        settings: 箱と、Genesis の衝突候補の上限。
        num_envs: 並列環境の数。正の整数。

    Raises:
        ValueError: num_envs が 1 未満。
        SimulationSetupError: Genesis がシーンを構築できなかった（アセットの読み込み失敗、設定値の不整合など）。
    """
    # Genesis は n_envs=0 のとき並列環境の次元を持たない形でテンソルを返すため、常に 1 以上で構築する
    if num_envs < 1:
        raise ValueError(f"num_envs は 1 以上が必要: {num_envs}")
    try:
        return _create_and_build_scene(gait_config, settings, num_envs)
    except gs.GenesisException as error:
        raise SimulationSetupError(f"シーン S1_push を構築できない: {error}") from error


def _create_and_build_scene(gait_config: GaitConfig, settings: PushSceneSettings, num_envs: int) -> PushScene:
    env_config = gait_config.env
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_config.control_dt_s, substeps=env_config.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=ENABLE_SELF_COLLISION,
            max_collision_pairs=settings.genesis.max_collision_pairs,
            friction_cone=getattr(gs.friction_cone, settings.genesis.friction_cone),
        ),
        show_viewer=False,
    )
    robot_config = gait_config.robot
    # 床と Go2 の摩擦は指定しない。歩行の学習時と同じ Genesis の既定値を使うため
    ground = scene.add_entity(gs.morphs.URDF(file=PLANE_URDF, fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file=GO2_URDF, pos=robot_config.base_init_pos_m, quat=robot_config.base_init_quat_wxyz)
    )
    box_config = settings.box
    box_x, box_y = box_config.initial_xy_m
    box_half_height = box_config.size_m[2] / 2
    box = scene.add_entity(
        gs.morphs.Box(size=box_config.size_m, pos=(box_x, box_y, box_half_height)),
        material=gs.materials.Rigid(rho=box_config.density_kg_per_m3(), friction=box_config.friction),
    )
    scene.build(n_envs=num_envs)
    return PushScene(ground_and_robot=FlatScene(scene=scene, robot=robot, camera=None), ground=ground, box=box)
