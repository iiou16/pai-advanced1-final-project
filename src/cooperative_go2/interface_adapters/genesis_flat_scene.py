"""シーン S0_flat（平面 ＋ Go2 ×1）を Genesis で構築する。物理ステップを進めるのはこのシーンだけ。"""

from dataclasses import dataclass

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.vis.camera import Camera

from cooperative_go2.entities.gait_config import GaitConfig

PLANE_URDF = "urdf/plane/plane.urdf"
GO2_URDF = "urdf/go2/urdf/go2.urdf"
# 1 台の歩行では脚同士の接触を扱わなくても歩容が成り立つため、自己衝突は計算しない（衝突候補と計算量を減らす）
ENABLE_SELF_COLLISION = False
# 評価動画は環境 0 の 1 台だけを撮るので、描画する環境を 0 に限る
RENDERED_ENV_INDICES = (0,)


class SimulationSetupError(RuntimeError):
    """Genesis のシーンを構築できなかったことを表す（アセットの読み込み失敗、設定値の不整合など）。"""


class CameraNotConfiguredError(RuntimeError):
    """カメラを置かずに構築したシーンで、カメラを使おうとしたことを表す。"""


@dataclass(frozen=True)
class CameraSpec:
    """評価動画を撮るオフスクリーンカメラ。カメラは環境 0 の Go2 を追従する。

    Attributes:
        resolution: 画像の幅と高さ [px]。
        offset_m: Go2 の胴体から見たカメラの相対位置 (x, y, z) [m]。
        fov_deg: 垂直方向の画角 [deg]。
    """

    resolution: tuple[int, int]
    offset_m: tuple[float, float, float]
    fov_deg: float


@dataclass(frozen=True)
class FlatScene:
    """構築済みのシーン S0_flat。

    Attributes:
        scene: Genesis のシーン。
        robot: Go2 のエンティティ。
        camera: 評価動画用のカメラ。カメラ無しで構築したら None。
    """

    scene: gs.Scene
    robot: RigidEntity
    camera: Camera | None

    def step(self) -> None:
        """物理を 1 制御周期進める。"""
        self.scene.step()

    def require_camera(self) -> Camera:
        """評価動画用のカメラを返す。

        Raises:
            CameraNotConfiguredError: カメラ無しで構築したシーンだった。
        """
        if self.camera is None:
            raise CameraNotConfiguredError("評価動画を撮るには CameraSpec を渡してシーンを構築する必要がある")
        return self.camera

    def diverged_mask(self) -> torch.Tensor:
        """物理計算が異常になった環境（NaN、衝突候補の上限超過など）を True とするマスク (num_envs,) を返す。"""
        return self.scene.rigid_solver.get_error_envs_mask()


def build_flat_scene(config: GaitConfig, camera_spec: CameraSpec | None) -> FlatScene:
    """平面と Go2 を置いたシーンを config.env.num_envs 個の並列環境で構築する。

    前提: gs.init() 済みであること。

    Args:
        config: 歩行の設定。env の物理パラメータと robot の初期位置を使う。
        camera_spec: 評価動画を撮る場合のカメラ。None ならカメラを置かない。

    Raises:
        SimulationSetupError: Genesis がシーンを構築できなかった。
    """
    env_config = config.env
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_config.control_dt_s, substeps=env_config.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=ENABLE_SELF_COLLISION, max_collision_pairs=env_config.max_collision_pairs
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=list(RENDERED_ENV_INDICES)),
        show_viewer=False,
    )
    robot_config = config.robot
    scene.add_entity(gs.morphs.URDF(file=PLANE_URDF, fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file=GO2_URDF, pos=robot_config.base_init_pos_m, quat=robot_config.base_init_quat_wxyz)
    )
    camera = None
    if camera_spec is not None:
        # 姿勢は build 前に add_camera で渡す。build 後の set_pose で位置と注視点だけを渡すと、
        # 上方向が既定のカメラ姿勢から引き継がれてカメラが傾く
        robot_pos = robot_config.base_init_pos_m
        camera = scene.add_camera(
            res=camera_spec.resolution,
            pos=tuple(p + o for p, o in zip(robot_pos, camera_spec.offset_m, strict=True)),
            lookat=robot_pos,
            fov=camera_spec.fov_deg,
            GUI=False,
        )
    try:
        scene.build(n_envs=env_config.num_envs)
    except gs.GenesisException as error:
        raise SimulationSetupError(f"シーン S0_flat を構築できない: {error}") from error
    if camera is not None:
        # build 時点のカメラとロボットの相対位置を保ったまま、向きを変えずに追従する
        camera.follow_entity(robot, fix_orientation=True)
    return FlatScene(scene=scene, robot=robot, camera=camera)
