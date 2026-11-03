"""1 台の箱押しの高レベル方策の観測ベクトルの構成。

学習済みの方策は観測の順序と正規化に依存する。順序を変えるとチェックポイントの互換性が失われる。
観測はすべてロボット座標系で表し、ロボットのワールド座標上の位置やヨー角そのものは含めない。
箱押しの結果はロボットと箱・目標の相対関係だけで決まり、絶対位置を与えると方策が場所に依存するため。
傾き（ロール・ピッチ）は本体座標系での重力方向として含め、転倒しかけた姿勢で無理な指令を出さないようにする。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.planar_frame import relative_xy_in_robot_frame, rotate_world_to_robot_xy, wrap_angle_rad
from cooperative_go2.entities.push_config import PushObservationScales

# 観測ベクトルを先頭から順に (成分名, 次元数) で並べたもの。ベクトル量はすべてロボット座標系
PUSH_OBSERVATION_LAYOUT: tuple[tuple[str, int], ...] = (
    ("robot_lin_vel_xy", 2),
    ("robot_yaw_rate", 1),
    ("robot_projected_gravity", 3),
    ("box_rel_pos_xy", 2),
    ("box_rel_yaw_sin_cos", 2),
    ("box_lin_vel_xy", 2),
    ("box_yaw_rate", 1),
    ("goal_rel_pos_xy", 2),
    ("box_contact", 1),
    ("last_command", 3),
)

PUSH_OBSERVATION_DIM = sum(size for _, size in PUSH_OBSERVATION_LAYOUT)


@dataclass(frozen=True)
class PlanarBodyState:
    """水平面で見た剛体（ロボットの胴体または箱）の状態。テンソルは並列環境のバッチ。

    Attributes:
        position_xy_m: 位置 (num_envs, 2)、ワールド座標系 [m]。
        yaw_rad: ヨー角 (num_envs,)。
        lin_vel_xy_mps: 線速度の水平成分 (num_envs, 2)、ワールド座標系 [m/s]。
        yaw_rate_radps: ヨー角速度 (num_envs,) [rad/s]。
    """

    position_xy_m: torch.Tensor
    yaw_rad: torch.Tensor
    lin_vel_xy_mps: torch.Tensor
    yaw_rate_radps: torch.Tensor


@dataclass(frozen=True)
class PushTaskSnapshot:
    """ある時点の Go2・箱・目標の位置（各環境の原点から見たワールド座標系の水平面）。

    Attributes:
        robot: Go2 の胴体の水平面の状態。
        box: 箱の水平面の状態。
        goal_xy_m: 目標領域の中心 (num_envs, 2) [m]。
    """

    robot: PlanarBodyState
    box: PlanarBodyState
    goal_xy_m: torch.Tensor


@dataclass(frozen=True)
class PushObservationInputs:
    """観測の材料。

    Attributes:
        robot: ロボットの胴体の状態。
        robot_projected_gravity: ロボットの本体座標系で表した重力方向の単位ベクトル (num_envs, 3)。
            歩行方策の観測（Go2Proprioception.projected_gravity）と同じ定義で、水平に立っていれば (0, 0, -1)。
        box: 箱の状態。
        goal_xy_m: 目標領域の中心 (num_envs, 2)、ワールド座標系 [m]。
        box_contact: ロボットが箱に接触している環境を True とするマスク (num_envs,)。
        last_commands: 直前に出した速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
            エピソードの最初は 0。
    """

    robot: PlanarBodyState
    robot_projected_gravity: torch.Tensor
    box: PlanarBodyState
    goal_xy_m: torch.Tensor
    box_contact: torch.Tensor
    last_commands: torch.Tensor


def build_push_observation(inputs: PushObservationInputs, scales: PushObservationScales) -> torch.Tensor:
    """PUSH_OBSERVATION_LAYOUT の順に正規化した観測 (num_envs, PUSH_OBSERVATION_DIM) を作る。

    箱の向きは、ロボットから見た相対ヨー角の sin と cos で表す。角度を直接入れると ±π で値が飛ぶため。

    Args:
        inputs: 観測の材料。
        scales: 成分ごとの正規化係数。速度指令の vx・vy には lin_vel、yaw_rate には ang_vel を掛ける。
            重力方向はすでに単位ベクトルなので係数を掛けない。
    """
    robot = inputs.robot
    box = inputs.box
    box_rel_yaw = wrap_angle_rad(box.yaw_rad - robot.yaw_rad)
    command_scales = torch.tensor(
        [scales.lin_vel, scales.lin_vel, scales.ang_vel],
        dtype=inputs.last_commands.dtype,
        device=inputs.last_commands.device,
    )
    return torch.cat(
        (
            rotate_world_to_robot_xy(robot.lin_vel_xy_mps, robot.yaw_rad) * scales.lin_vel,
            robot.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
            inputs.robot_projected_gravity,
            relative_xy_in_robot_frame(box.position_xy_m, robot.position_xy_m, robot.yaw_rad) * scales.position,
            torch.stack((torch.sin(box_rel_yaw), torch.cos(box_rel_yaw)), dim=1),
            rotate_world_to_robot_xy(box.lin_vel_xy_mps, robot.yaw_rad) * scales.lin_vel,
            box.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
            relative_xy_in_robot_frame(inputs.goal_xy_m, robot.position_xy_m, robot.yaw_rad) * scales.position,
            inputs.box_contact.to(inputs.last_commands.dtype).unsqueeze(1),
            inputs.last_commands * command_scales,
        ),
        dim=-1,
    )
