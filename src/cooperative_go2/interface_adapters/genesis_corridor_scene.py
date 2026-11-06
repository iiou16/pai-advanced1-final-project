"""通路のシーン（S2_corridor・S3_team）を Genesis で構築する。

床・Go2 を設定の台数・通路の両側の壁・通路をふさぐ箱を置く。床と Go2 の置き方・物理の時間刻み・摩擦は
歩行の学習時のシーン（S0_flat）と同じにする。固定した歩行方策を、学習したときと同じ条件で動かすため。
"""

import dataclasses
from dataclasses import dataclass

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.vis.camera import Camera

from cooperative_go2.entities.corridor_scene_config import CorridorSceneConfig
from cooperative_go2.entities.gait_config import GaitConfig, Go2RobotConfig
from cooperative_go2.interface_adapters.config_loader import CorridorSceneSettings
from cooperative_go2.interface_adapters.genesis_flat_scene import (
    ENABLE_SELF_COLLISION,
    GO2_URDF,
    PLANE_URDF,
    RENDERED_ENV_INDICES,
    FlatScene,
)
from cooperative_go2.interface_adapters.genesis_push_scene import (
    CONTACT_VALID_MASK_KEY,
    GOAL_MARKER_COLOR_RGB,
    GOAL_MARKER_THICKNESS_M,
)
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

VERTICAL_AXIS_INDEX = 2
# 壁は灰色にして、箱・床と見分けやすくする
WALL_COLOR_RGB = (0.55, 0.55, 0.6)
# 通路の両側（+y 側・-y 側）
WALL_SIDES = (1.0, -1.0)


@dataclass(frozen=True)
class FixedCameraSpec:
    """評価動画を撮る、シーンに固定したオフスクリーンカメラ。通路の全体を 1 つの画角で撮るため、ロボットを追わない。

    Attributes:
        resolution: 画像の幅と高さ [px]。
        position_m: カメラの位置 (x, y, z) [m]。ワールド座標系。
        lookat_m: 注視点 (x, y, z) [m]。ワールド座標系。
        fov_deg: 垂直方向の画角 [deg]。
    """

    resolution: tuple[int, int]
    position_m: tuple[float, float, float]
    lookat_m: tuple[float, float, float]
    fov_deg: float


@dataclass(frozen=True)
class CorridorSceneRequest:
    """通路のシーンの作り方。

    Attributes:
        num_envs: 並列環境の数。正の整数。
        camera: 評価動画を撮るならそのカメラ（退避領域を示す円板も置く）。撮らないなら None。
    """

    num_envs: int
    camera: FixedCameraSpec | None


@dataclass(frozen=True)
class CorridorScene:
    """構築済みの通路のシーン。

    Attributes:
        stepper: 物理ステップと、容量の超過・NaN の検出に使う（床と 1 台目の Go2 を持つ FlatScene）。
        robots: Go2 のエンティティ。設定の robots と同じ順。
        robot_configs: 各 Go2 の初期位置を反映した、歩行方策の学習時のロボットの設定。robots と同じ順。
        box: 箱のエンティティ。動的な剛体で、Go2・壁との接触と床との摩擦だけで動く。
        walls: 通路の両側の壁（固定）。
        camera: 評価動画用のカメラ。カメラ無しで構築したら None。
    """

    stepper: FlatScene
    robots: tuple[RigidEntity, ...]
    robot_configs: tuple[Go2RobotConfig, ...]
    box: RigidEntity
    walls: tuple[RigidEntity, ...]
    camera: Camera | None

    def step(self) -> None:
        """物理を 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: いずれかの環境で衝突候補・接触点の数が上限を超えた。
        """
        self.stepper.step()

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで NaN が出た（物理計算が発散した）環境を True とするマスク (num_envs,) を返す。"""
        return self.stepper.diverged_mask()

    def robot_box_contact_mask(self, robot_index: int) -> torch.Tensor:
        """robot_index 番目の Go2 のいずれかのリンクが箱に触れている環境のマスク (num_envs,) を返す。"""
        contacts = self.robots[robot_index].get_contacts(with_entity=self.box)
        return contacts[CONTACT_VALID_MASK_KEY].any(dim=-1)


def build_corridor_scene(
    gait_config: GaitConfig, settings: CorridorSceneSettings, request: CorridorSceneRequest
) -> CorridorScene:
    """床・Go2・壁・箱を置いた通路のシーンを request.num_envs 個の並列環境で構築する。

    前提: gs.init() 済みであること。

    Args:
        gait_config: 歩行方策を学習したときの設定。制御周期・サブステップ数と Go2 の初期の高さ・姿勢を使う。
        settings: 通路のシーンの配置と、Genesis の接触の設定。
        request: 並列環境の数と、評価動画を撮るかどうか。

    Raises:
        ValueError: num_envs が 1 未満。
        SimulationSetupError: Genesis がシーンを構築できなかった（アセットの読み込み失敗、設定値の不整合など）。
    """
    # Genesis は n_envs=0 のとき並列環境の次元を持たない形でテンソルを返すため、常に 1 以上で構築する
    if request.num_envs < 1:
        raise ValueError(f"num_envs は 1 以上が必要: {request.num_envs}")
    try:
        return _create_and_build_scene(gait_config, settings, request)
    except gs.GenesisException as error:
        raise SimulationSetupError(f"通路のシーンを構築できない: {error}") from error


def robot_configs_for_placements(gait_config: GaitConfig, scene: CorridorSceneConfig) -> tuple[Go2RobotConfig, ...]:
    """各 Go2 の初期位置（水平面）を設定の配置に置き換えた、ロボットの設定を返す。高さと姿勢は学習時のまま。"""
    robot_config = gait_config.robot
    base_z_m = robot_config.base_init_pos_m[VERTICAL_AXIS_INDEX]
    return tuple(
        dataclasses.replace(robot_config, base_init_pos_m=(*placement.initial_xy_m, base_z_m))
        for placement in scene.robots
    )


def _create_and_build_scene(
    gait_config: GaitConfig, settings: CorridorSceneSettings, request: CorridorSceneRequest
) -> CorridorScene:
    env_config = gait_config.env
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_config.control_dt_s, substeps=env_config.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=ENABLE_SELF_COLLISION,
            max_collision_pairs=settings.genesis.max_collision_pairs,
            friction_cone=getattr(gs.friction_cone, settings.genesis.friction_cone),
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=list(RENDERED_ENV_INDICES)),
        show_viewer=False,
    )
    layout = settings.scene
    # 床・Go2・壁の摩擦は指定しない。歩行の学習時と同じ Genesis の既定値を使うため
    scene.add_entity(gs.morphs.URDF(file=PLANE_URDF, fixed=True))
    robot_configs = robot_configs_for_placements(gait_config, layout)
    robots = tuple(
        scene.add_entity(gs.morphs.URDF(file=GO2_URDF, pos=config.base_init_pos_m, quat=config.base_init_quat_wxyz))
        for config in robot_configs
    )
    walls = _add_walls(scene, layout)
    box_config = layout.box
    box_x, box_y = box_config.initial_xy_m
    box = scene.add_entity(
        gs.morphs.Box(size=box_config.size_m, pos=(box_x, box_y, box_config.size_m[VERTICAL_AXIS_INDEX] / 2)),
        material=gs.materials.Rigid(rho=box_config.density_kg_per_m3(), friction=box_config.friction),
    )
    camera = None
    if request.camera is not None:
        _add_retreat_marker(scene, layout)
        spec = request.camera
        camera = scene.add_camera(
            res=spec.resolution, pos=spec.position_m, lookat=spec.lookat_m, fov=spec.fov_deg, GUI=False
        )
    scene.build(n_envs=request.num_envs)
    return CorridorScene(
        stepper=FlatScene(scene=scene, robot=robots[0], camera=camera),
        robots=robots,
        robot_configs=robot_configs,
        box=box,
        walls=walls,
        camera=camera,
    )


def _add_walls(scene: gs.Scene, layout: CorridorSceneConfig) -> tuple[RigidEntity, ...]:
    corridor = layout.corridor
    walls_config = layout.walls
    length_m = corridor.exit_x_m - corridor.entrance_x_m
    thickness_m = walls_config.outer_half_extent_y_m - corridor.width_m / 2
    center_x_m = (corridor.entrance_x_m + corridor.exit_x_m) / 2
    offset_y_m = corridor.width_m / 2 + thickness_m / 2
    return tuple(
        scene.add_entity(
            gs.morphs.Box(
                size=(length_m, thickness_m, walls_config.height_m),
                pos=(center_x_m, corridor.center_y_m + side * offset_y_m, walls_config.height_m / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=WALL_COLOR_RGB),
        )
        for side in WALL_SIDES
    )


def _add_retreat_marker(scene: gs.Scene, layout: CorridorSceneConfig) -> None:
    retreat = layout.retreat_area
    center_x, center_y = retreat.center_xy_m
    scene.add_entity(
        gs.morphs.Cylinder(
            radius=retreat.radius_m,
            height=GOAL_MARKER_THICKNESS_M,
            pos=(center_x, center_y, GOAL_MARKER_THICKNESS_M / 2),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=GOAL_MARKER_COLOR_RGB),
    )
