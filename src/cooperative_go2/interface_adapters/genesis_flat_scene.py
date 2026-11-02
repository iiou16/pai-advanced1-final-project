"""シーン S0_flat（平面 ＋ Go2 ×1）を Genesis で構築し、物理ステップと物理計算の異常の検出を担う。"""

from dataclasses import dataclass

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_to_torch
from genesis.vis.camera import Camera

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

PLANE_URDF = "urdf/plane/plane.urdf"
GO2_URDF = "urdf/go2/urdf/go2.urdf"
# 1 台の歩行では脚同士の接触を扱わなくても歩容が成り立つため、自己衝突は計算しない（衝突候補と計算量を減らす）
ENABLE_SELF_COLLISION = False
# 評価動画は環境 0 の 1 台だけを撮るので、描画する環境を 0 に限る
RENDERED_ENV_INDICES = (0,)
# Genesis が環境ごとに立てるエラーのビット。容量の超過は設定の不備、NaN は物理計算の発散として扱いを分ける
CAPACITY_OVERFLOW_BITS = (
    ErrorCode.OVERFLOW_COLLISION_PAIRS | ErrorCode.OVERFLOW_CANDIDATE_CONTACTS | ErrorCode.OVERFLOW_CONTACTS
)
NAN_BITS = ErrorCode.INVALID_CONTACT_NAN | ErrorCode.INVALID_FORCE_NAN | ErrorCode.INVALID_ACC_NAN


class SimulationCapacityExceededError(SimulationSetupError):
    """衝突候補や接触点の数が、設定した上限（env.max_collision_pairs など）を超えたことを表す。"""


class CameraNotConfiguredError(RuntimeError):
    """カメラを置かずに構築したシーンで、カメラを使おうとしたことを表す。"""


@dataclass(frozen=True)
class CameraSpec:
    """評価動画を撮るオフスクリーンカメラ。カメラは環境 0 の Go2 を、向きを変えずに追従する。

    Attributes:
        resolution: 画像の幅と高さ [px]。
        offset_m: Go2 の胴体の初期位置からカメラまでの相対位置 (x, y, z) [m]。ワールド座標系の軸で測る。
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

    def require_camera(self) -> Camera:
        """評価動画用のカメラを返す。

        Raises:
            CameraNotConfiguredError: カメラ無しで構築したシーンだった。
        """
        if self.camera is None:
            raise CameraNotConfiguredError("評価動画を撮るには CameraSpec を渡してシーンを構築する必要がある")
        return self.camera

    def step(self) -> None:
        """物理を 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: いずれかの環境で衝突候補・接触点の数が上限を超えた。
        """
        self.scene.step()
        overflowed = (self._error_bits() & CAPACITY_OVERFLOW_BITS) != 0
        if bool(overflowed.any()):
            raise SimulationCapacityExceededError(
                f"衝突候補・接触点の数が上限を超えた環境がある: {torch.nonzero(overflowed).flatten().tolist()}。"
                "env.max_collision_pairs を見直す"
            )

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで NaN が出た（物理計算が発散した）環境を True とするマスク (num_envs,) を返す。"""
        return (self._error_bits() & NAN_BITS) != 0

    def _error_bits(self) -> torch.Tensor:
        # Genesis の公開 API（get_error_envs_mask）はエラーの種類を区別しないため、版を固定した上で内部の errno を読む
        return qd_to_torch(self.scene.rigid_solver._errno)


def build_flat_scene(config: GaitConfig, camera_spec: CameraSpec | None) -> FlatScene:
    """平面と Go2 を置いたシーンを config.env.num_envs 個の並列環境で構築する。

    前提: gs.init() 済みであること。

    Args:
        config: 歩行の設定。env の物理パラメータと robot の初期位置を使う。
        camera_spec: 評価動画を撮る場合のカメラ。None ならカメラを置かない。

    Raises:
        SimulationSetupError: Genesis がシーンを構築できなかった（アセットの読み込み失敗、設定値の不整合など）。
    """
    try:
        flat_scene = _create_and_build_scene(config, camera_spec)
        if flat_scene.camera is not None:
            # build 時点のカメラとロボットの相対位置を保ったまま、向きを変えずに追従する
            flat_scene.camera.follow_entity(flat_scene.robot, fix_orientation=True)
    except gs.GenesisException as error:
        raise SimulationSetupError(f"シーン S0_flat を構築できない: {error}") from error
    return flat_scene


def _create_and_build_scene(config: GaitConfig, camera_spec: CameraSpec | None) -> FlatScene:
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
            pos=tuple(position + offset for position, offset in zip(robot_pos, camera_spec.offset_m, strict=True)),
            lookat=robot_pos,
            fov=camera_spec.fov_deg,
            GUI=False,
        )
    scene.build(n_envs=env_config.num_envs)
    return FlatScene(scene=scene, robot=robot, camera=camera)
